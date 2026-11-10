"""One real codec/detector episode, for training or frozen paired Validation."""
from dataclasses import asdict
import json
from pathlib import Path
import time

from agent_v4.checkpoint import load
from agent_v4.codec import encode, require_zero_identity, vanilla_encode, require_runtime_profile
from agent_v4.reward import RewardConfig, join, native_decision_rows
from agent_v4.runtime import verify_source
from au_rate_audit import read_au_bits
from decoder_alignment import verify_decode_mapping
from full_train import atomic_json
from real_reward_pilot import (preprocess_mov, dataset_source, detect_all,
    load_frozen_yolov5, read_lock_manifest, read_baseline_config,
    convert_10bit_to_8bit, run_cmd, sha256_file)
from reward_join import read_detections


def decode_detect(decoder, detector, stream, out, frames, vid, threads):
    dec10, dec8 = out/"decoded_10bit.yuv", out/"decoded_8bit.yuv"
    run_cmd([str(decoder), "-b", str(stream), "-o", str(dec10), "--threads", str(threads)],
            out/"decoder.log", timeout=900)
    mapping = verify_decode_mapping(stream, dec10, decoder, frames, threads)
    convert_10bit_to_8bit(dec10, dec8, frames)
    det = out/"decoded_detections.jsonl"
    detect_all(detector, dec8, frames, vid, det)
    dec10.unlink()
    dec8.unlink()
    return read_detections(det, vid, set(range(frames))), mapping


def collect(repo, decoder, checkpoint, output, input_root, split, index, frames,
            start, rate, seed, deterministic=False, paired=False, zero_gate=False,
            stochastic_validation=False):
    ckpt, model, _ = load(checkpoint)
    exp = ckpt["experiment"]
    require_runtime_profile(exp)
    verify_source(repo,exp)
    splits, records = read_lock_manifest(input_root)
    lock = read_baseline_config(input_root)
    import hashlib
    split_sha = hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest()
    if split_sha != exp["split_ids_sha256"]:
        raise ValueError("Collector split IDs differ from experiment")
    if split not in ("train", "val") or not 0 <= index < len(splits[split]):
        raise ValueError("Collector accepts locked Train/Val only; never Test")
    if split == "val" and (deterministic == stochastic_validation or not paired or start != 0):
        raise ValueError("Validation requires exactly one declared greedy/sampled rule, paired, frame0")
    if split == "train" and (deterministic or stochastic_validation):
        raise ValueError("Training requires stochastic fresh on-policy rollouts")
    if split == 'train' and exp['ppo_config'].get('paired_baseline') and not paired:
        raise ValueError("This experiment requires paired vanilla for every Train clip")
    if zero_gate and not paired:
        raise ValueError("Zero-action identity gate requires a paired Validation episode")
    encoder = repo/"bin/release-static/vvencapp"
    if any(sha256_file(p) != exp[key] for p,key in
           ((encoder,"encoder_sha256"),(decoder,"decoder_sha256"))):
        raise ValueError("Codec differs from experiment")
    if lock["lock_sha256"] != exp["baseline_lock_sha256"]:
        raise ValueError("Baseline protocol changed")
    from decoder_alignment import require_auditor
    if sha256_file(require_auditor(decoder)) != exp["auditor_sha256"]:
        raise ValueError("POC auditor differs from experiment")
    vid = splits[split][index]
    if vid in splits["test"] or vid in splits["val" if split == "train" else "train"]:
        raise ValueError("Split violation")
    checkpoint_sha = sha256_file(checkpoint)
    began = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    source = output/"reference_8bit.yuv"
    preprocessing = preprocess_mov(dataset_source(records, vid), source, frames,
                                  output/"source_ffmpeg.log", start_frame=start)
    source_sha = sha256_file(source)
    stream, rows, aus = encode(repo, encoder, source, output/"online", frames,
        checkpoint, rate, seed, deterministic, exp["encoder_threads"],evaluation=split == 'val')
    detector, detector_provenance = load_frozen_yolov5(repo, lock)
    source_det = output/"source_detections.jsonl"
    detect_all(detector, source, frames, vid, source_det)
    source_boxes = read_detections(source_det, vid, set(range(frames)))
    decoded_boxes, mapping = decode_detect(decoder, detector, stream, output/"online",
        frames, vid, exp["decoder_threads"])
    reward_config = RewardConfig(**exp["reward"])
    joined, metrics = join(rows, aus, source_boxes, decoded_boxes, rate*1000, reward_config, vid)
    gamma = ckpt["ppo_config"]["gamma"]
    metrics["selection_score"] = sum(gamma**i*r["reward"] for i,r in enumerate(joined))
    with (output/"rollout.jsonl").open("w", encoding="utf-8") as f:
        for row in joined:
            f.write(json.dumps(row, allow_nan=False, separators=(",", ":"))+"\n")
    actions = [a for r in rows for a in r["action"]]
    summary = {"split": split, "index": index, "video_id": vid, "frames": frames,
        "start_frame": start, "rate": rate, "seed": seed, "deterministic": deterministic,
        "policy_sha256": checkpoint_sha, "experiment": exp,
        "source_sha256": source_sha, "preprocessing": preprocessing,
        "detector": detector_provenance, "alignment": mapping,
        "metrics": metrics, "rollout_sha256": sha256_file(output/"rollout.jsonl"),
        "action_histogram": {str(i):actions.count(i) for i in range(-2, 3)}}
    if paired:
        base = vanilla_encode(encoder, source, output/"vanilla", frames, rate, exp["encoder_threads"])
        base_boxes, base_mapping = decode_detect(decoder, detector, base, output/"vanilla", frames,
                                                 vid, exp["decoder_threads"])
        base_order = native_decision_rows(output/"vanilla"/"states.txt",frames)
        base_rows, base_metrics = join(base_order, read_au_bits(output/"vanilla"/"au_bits.csv", base, frames),
                               source_boxes, base_boxes, rate*1000, reward_config, vid)
        base_metrics["selection_score"] = sum(gamma**i*r["reward"] for i,r in enumerate(base_rows))
        summary.update(vanilla=base_metrics, vanilla_alignment=base_mapping,
                       delta_f1=metrics["f1"]-base_metrics["f1"],
                       delta_kbps=metrics["actual_kbps"]-base_metrics["actual_kbps"],
                       delta_score=metrics["selection_score"]-base_metrics["selection_score"])
        # A zero-map result must reproduce the baseline on EVERY pair, not
        # only the first zero-gate pair. Fail before publishing a success.
        summary['zero_action_byte_identity'] = require_zero_identity(stream,base,rows,metrics,base_metrics,output)
        if zero_gate:
            zero, _, _ = encode(repo, encoder, source, output/"zero_gate", frames,
                checkpoint, rate, seed, True, exp["encoder_threads"], zero=True)
            if sha256_file(zero) != sha256_file(base):
                raise ValueError("Zero action bitstream != VVenC baseline")
            summary["zero_action_byte_identity"] = True
    if sha256_file(checkpoint) != checkpoint_sha or sha256_file(source) != source_sha:
        raise ValueError("Policy/source modified while collecting")
    summary["elapsed_seconds"] = time.monotonic()-began
    atomic_json(output/"summary.json", summary)
    print("AGENT_V4_REAL_EPISODE_PASS " + json.dumps({"video_id": vid, "metrics": metrics,
          "policy_sha256": checkpoint_sha, "elapsed": summary["elapsed_seconds"]}), flush=True)
    return summary

