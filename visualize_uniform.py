"""
Visualize the generation process step by step
Shows masked diffusion and uniform diffusion side-by-side
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation

import diffusion
import uniform


def generate_diffusion_frames(
    model,
    num_blocks=5,
    prompt_len=16,
    temp=0.8,
    confidence_threshold=0.95,
    top_k=2,
):
    """
    Generate samples and capture each denoising step for masked diffusion

    Args:
        model: The trained diffusion model
        num_blocks: Number of blocks to generate
        prompt_len: Length of initial prompt
        temp: Sampling temperature
        confidence_threshold: Confidence threshold for decoding
        top_k: Top-k sampling parameter

    Returns:
        List of (all_tokens, mask_for_all, block_idx) tuples for each frame
    """
    device = next(model.parameters()).device
    block_size = diffusion.block_size
    mask_token_id = diffusion.mask_token_id

    print(f"Pre-calculating {num_blocks} blocks for masked diffusion...")

    all_frames = []
    all_tokens_history = diffusion.data[:prompt_len].tolist()

    def capture_frame(all_tokens_history, x, masked, block_idx):
        """Build and capture full sequence frame"""
        full_tokens = torch.tensor(all_tokens_history, dtype=torch.long, device=device)
        full_tokens = torch.cat(
            [full_tokens, x[0, prompt_len : prompt_len + block_len]]
        )
        full_mask = torch.zeros(len(full_tokens), dtype=torch.bool, device=device)
        full_mask[len(all_tokens_history) :] = masked[
            0, prompt_len : prompt_len + block_len
        ]
        all_frames.append(
            (full_tokens.cpu().clone(), full_mask.cpu().clone(), block_idx)
        )

    for block_idx in range(num_blocks):
        # How many tokens to generate this block
        max_new_tokens = 240
        block_len = min(block_size - prompt_len, max_new_tokens)

        # Initialize: last prompt_len tokens + masks
        x = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=device)
        x[0, :prompt_len] = torch.tensor(
            all_tokens_history[-prompt_len:], device=device
        )

        # Track which positions need decoding
        masked = torch.zeros(1, block_size, dtype=torch.bool, device=device)
        masked[0, prompt_len : prompt_len + block_len] = True

        capture_frame(all_tokens_history, x, masked, block_idx)

        # Iteratively decode
        step = 0
        while masked.any():
            # Get predictions and confidences
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
            confidences = top_k_probs.sum(dim=-1)

            # Decode high-confidence masked positions (or at least 1)
            decode_mask = (confidences >= confidence_threshold) & masked
            if not decode_mask.any():
                masked_confidences = torch.where(
                    masked, confidences, torch.tensor(-float("inf"), device=device)
                )
                decode_mask.view(-1)[masked_confidences.argmax()] = True

            # Sample from top-k and update
            top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(
                1, block_size
            )
            sampled_tokens = torch.gather(
                top_k_indices, -1, sampled_k.unsqueeze(-1)
            ).squeeze(-1)

            x = torch.where(decode_mask, sampled_tokens, x)
            masked = masked & ~decode_mask

            capture_frame(all_tokens_history, x, masked, block_idx)
            step += 1

        # Extract and append generated tokens for next block
        all_tokens_history.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    print(f"Masked Diffusion: Generated {len(all_frames)} frames")
    return all_frames


def generate_uniform_frames(model, num_blocks=5, prompt_len=16, num_steps=64, temp=1.0):
    """
    Generate samples and capture each refinement step for uniform diffusion

    Args:
        model: The trained uniform diffusion model
        num_blocks: Number of blocks to generate
        prompt_len: Length of initial prompt
        num_steps: Number of refinement steps per block
        temp: Sampling temperature

    Returns:
        List of (all_tokens, block_idx, step_idx) tuples for each frame
    """
    device = next(model.parameters()).device
    block_size = uniform.block_size
    vocab_size = uniform.vocab_size

    print(f"Pre-calculating {num_blocks} blocks for uniform diffusion...")

    all_frames = []
    all_tokens_history = uniform.data[:prompt_len].tolist()

    def capture_frame(all_tokens_history, x, block_idx, step_idx):
        """Build and capture full sequence frame"""
        full_tokens = torch.tensor(all_tokens_history, dtype=torch.long, device=device)
        full_tokens = torch.cat(
            [full_tokens, x[0, prompt_len : prompt_len + block_len]]
        )
        all_frames.append((full_tokens.cpu().clone(), block_idx, step_idx))

    for block_idx in range(num_blocks):
        # How many tokens to generate this block
        max_new_tokens = 240
        block_len = min(block_size - prompt_len, max_new_tokens)

        # Initialize: last prompt_len tokens + random tokens for positions to generate
        x = torch.randint(
            0, vocab_size, (1, block_size), dtype=torch.long, device=device
        )
        x[0, :prompt_len] = torch.tensor(
            all_tokens_history[-prompt_len:], device=device
        )

        # Capture initial random state
        capture_frame(all_tokens_history, x, block_idx, 0)

        # Iteratively refine: run model multiple times, each time sampling tokens
        for step in range(num_steps):
            # Get predictions for all positions
            logits, _ = model(x)
            probs = F.softmax(logits / temp, dim=-1)

            # Sample tokens for each position after prompt
            sampled = torch.multinomial(probs.view(-1, vocab_size), 1).view(
                1, block_size
            )
            x[0, prompt_len:] = sampled[0, prompt_len:]

            # Capture frame
            capture_frame(all_tokens_history, x, block_idx, step + 1)

        # Extract and append generated tokens for next block
        all_tokens_history.extend(x[0, prompt_len : prompt_len + block_len].tolist())

    print(f"Uniform Diffusion: Generated {len(all_frames)} frames")
    return all_frames


def escape_latex_chars(text):
    """Escape special LaTeX characters to prevent matplotlib mathtext parsing"""
    # Characters that trigger mathtext parsing in matplotlib
    special_chars = {
        "$": r"\$",
        "_": r"\_",
        "^": r"\^",
        "\\": r"\\",
        "{": r"\{",
        "}": r"\}",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "~": r"\~",
    }
    for char, escaped in special_chars.items():
        text = text.replace(char, escaped)
    return text


def animate_comparison(diffusion_frames, uniform_frames, num_blocks, chars_per_row=64):
    """Create animation comparing masked diffusion and uniform diffusion"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    text_obj1 = ax1.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        linespacing=1.2,
        multialignment="left",
    )

    text_obj2 = ax2.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=10,
        family="monospace",
        linespacing=1.2,
        multialignment="left",
    )

    def update(frame_idx):
        # Update uniform diffusion (left)
        uniform_idx = min(frame_idx, len(uniform_frames) - 1)
        frame_tokens, block_idx, step_idx = uniform_frames[uniform_idx]

        text_chars = [
            uniform.decode([t.item()]) if uniform.decode([t.item()]) != "\n" else " "
            for t in frame_tokens
        ]
        continuous_text = "".join(text_chars)
        continuous_text = escape_latex_chars(continuous_text)
        lines = [
            continuous_text[i : i + chars_per_row]
            for i in range(0, len(continuous_text), chars_per_row)
        ]
        text_obj1.set_text("\n".join(lines))

        ax1.set_title(
            f"Uniform Diffusion - Block {block_idx + 1}/{num_blocks} - Step {step_idx}/128",
            fontsize=14,
            pad=-20,
            y=0.98,
        )

        # Update masked diffusion (right)
        diffusion_idx = min(frame_idx, len(diffusion_frames) - 1)
        frame_tokens, mask, block_idx = diffusion_frames[diffusion_idx]

        text_chars = []
        for idx in range(len(frame_tokens)):
            char = diffusion.decode([frame_tokens[idx].item()])
            if char == "\n":
                char = " "
            text_chars.append("█" if mask[idx] else char)

        continuous_text = "".join(text_chars)
        continuous_text = escape_latex_chars(continuous_text)
        lines = [
            continuous_text[i : i + chars_per_row]
            for i in range(0, len(continuous_text), chars_per_row)
        ]
        text_obj2.set_text("\n".join(lines))

        num_masked = mask.sum().item()
        ax2.set_title(
            f"Masked Diffusion - Block {block_idx + 1}/{num_blocks} - Remaining: {num_masked}",
            fontsize=14,
            pad=-20,
            y=0.98,
        )

        return [text_obj1, text_obj2]

    # Add pause frames at the end (50 frames = 500ms pause)
    pause_frames = 50
    max_frames = max(len(diffusion_frames), len(uniform_frames)) + pause_frames

    anim = FuncAnimation(
        fig, update, frames=max_frames, interval=10, blit=False, repeat=True
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return anim


def main():
    parser = argparse.ArgumentParser(
        description="Visualize masked diffusion vs uniform diffusion generation"
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=3,
        help="Number of blocks to generate (default: 3)",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=16,
        help="Length of initial prompt (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for generation (default: 1337)",
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}\n")

    # Load and generate masked diffusion
    diffusion_path = os.path.join(os.path.dirname(__file__), "weights", "diffusion.pt")
    print(f"Loading masked diffusion model from {diffusion_path}...")
    diffusion_model = diffusion.Model().to(device)
    diffusion_model.load_state_dict(torch.load(diffusion_path, map_location=device))
    diffusion_model.eval()

    diffusion_frames = generate_diffusion_frames(
        diffusion_model, args.blocks, args.prompt_len, temp=0.7
    )

    # Load and generate uniform diffusion
    uniform_path = os.path.join(os.path.dirname(__file__), "weights", "uniform.pt")
    print(f"Loading uniform diffusion model from {uniform_path}...")
    uniform_model = uniform.Model().to(device)
    uniform_model.load_state_dict(torch.load(uniform_path, map_location=device))
    uniform_model.eval()

    uniform_frames = generate_uniform_frames(
        uniform_model, args.blocks, args.prompt_len, num_steps=128, temp=0.8
    )

    print("Done! Showing comparison animation...\n")
    anim = animate_comparison(diffusion_frames, uniform_frames, args.blocks)

    plt.show()


if __name__ == "__main__":
    main()
