#!/usr/bin/env python3
"""
Multi-GPU Task-Parallel Training for Replacement Blocks

This script manages training tasks across multiple GPUs where each GPU works on
a different block/linear_type combination simultaneously, maximizing hardware utilization.

Features:
- Assigns different training tasks to different GPUs
- Automatically schedules new tasks when GPUs become available
- Monitors GPU memory and availability
- Supports custom batch sizes and training parameters
- Comprehensive logging and progress tracking

Usage:
    python train_blocks_parallel.py \
        --data_dir data/af2rank_single/af2_block_data/ \
        --output_dir AFdistill/parallel_training \
        --blocks 1 2 3 4 5 \
        --linear_types full diagonal \
        --max_epochs 50 \
        --batch_size 8
"""

import argparse
import os
import sys
import time
import threading
import queue
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
from tqdm import tqdm

# Add openfold to path
sys.path.append(str(Path.home() / 'openfold'))


@dataclass
class TrainingTask:
    """Represents a single training task"""
    block_idx: int
    linear_type: str
    gpu_id: int
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    process: Optional[subprocess.Popen] = None
    progress_bar: Optional[tqdm] = None  # Progress bar for GPU 0 tasks
    
    @property
    def task_id(self) -> str:
        return f"block_{self.block_idx:02d}_{self.linear_type}"
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class GPUTaskScheduler:
    """Manages training tasks across multiple GPUs"""
    
    def __init__(self, args):
        self.args = args
        self.home_dir = Path.home()
        self.data_dir = self.home_dir / args.data_dir
        self.output_dir = self.home_dir / args.output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize GPU information
        self.num_gpus = self._get_num_gpus()
        print(f"Detected {self.num_gpus} GPUs")
        
        # Generate training tasks
        self.tasks = self._generate_tasks()
        print(f"Generated {len(self.tasks)} training tasks")
        
        # Task management
        self.task_queue = queue.Queue()
        self.running_tasks: Dict[int, TrainingTask] = {}  # gpu_id -> task
        self.completed_tasks: List[TrainingTask] = []
        self.failed_tasks: List[TrainingTask] = []
        
        # Populate task queue
        for task in self.tasks:
            self.task_queue.put(task)
        
        # Logging
        self.log_file = self.output_dir / "parallel_training.log"
        self.results_file = self.output_dir / "parallel_results.json"
        
    def _get_num_gpus(self) -> int:
        """Get number of available GPUs"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return len(result.stdout.strip().split('\n'))
            else:
                print("Warning: Could not detect GPUs, defaulting to 1")
                return 1
        except FileNotFoundError:
            print("Warning: nvidia-smi not found, defaulting to 1 GPU")
            return 1
    
    def _generate_tasks(self) -> List[TrainingTask]:
        """Generate all training tasks"""
        tasks = []
        task_id = 0
        
        for block_idx in self.args.blocks:
            for linear_type in self.args.linear_types:
                task = TrainingTask(
                    block_idx=block_idx,
                    linear_type=linear_type,
                    gpu_id=-1  # Will be assigned when scheduled
                )
                tasks.append(task)
                task_id += 1
        
        return tasks
    
    def _check_task_completion(self, task: TrainingTask) -> bool:
        """Check if a training task already exists"""
        checkpoint_dir = self.output_dir / f"block_{task.block_idx:02d}" / task.linear_type
        checkpoint_file = checkpoint_dir / "best_model.ckpt"
        return checkpoint_file.exists()
    
    def _build_training_command(self, task: TrainingTask) -> List[str]:
        """Build command for a single training task"""
        cmd = [
            sys.executable, "openfold/block_replacement_scripts/train_single_block_replacements.py",
            "--data_dir", str(self.args.data_dir),
            "--output_dir", str(self.args.output_dir),
            "--blocks", str(task.block_idx),
            "--linear_types", task.linear_type,
            "--batch_size", str(self.args.batch_size),
            "--max_epochs", str(self.args.max_epochs),
            "--learning_rate", str(self.args.learning_rate),
            "--weight_decay", str(self.args.weight_decay),
            "--val_fraction", str(self.args.val_fraction),
            "--num_workers", str(self.args.num_workers),
            "--distributed_backend", str(self.args.distributed_backend),
        ]
        
        if self.args.compile:
            cmd.append("--compile")

        if self.args.hidden_dim is not None:
            cmd.extend(["--hidden_dim", str(self.args.hidden_dim)])
        
        # Add wandb if enabled
        if self.args.wandb:
            cmd.extend(["--wandb"])
            cmd.extend(["--wandb_project", self.args.wandb_project])
            cmd.extend(["--wandb_entity", self.args.wandb_entity])
            cmd.extend(["--experiment_name", f"{self.args.experiment_name}_{task.task_id}"])
        
        return cmd
    
    def _run_task_on_gpu(self, task: TrainingTask, gpu_id: int) -> TrainingTask:
        """Run a single training task on specified GPU"""
        task.gpu_id = gpu_id
        task.status = "running"
        task.start_time = time.time()
        
        # Set GPU environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Build command
        cmd = self._build_training_command(task)
        
        # Log task start
        self._log(f"🚀 Starting {task.task_id} on GPU {gpu_id}")
        
        # Create progress bar for GPU 0 tasks
        if gpu_id == 0:
            task.progress_bar = tqdm(
                total=self.args.max_epochs,
                desc=f"GPU0: {task.task_id}",
                position=0,
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        try:
            # Run training
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(self.home_dir)
            )
            
            task.process = process
            
            # Monitor progress for GPU 0 tasks in a separate thread
            progress_thread = None
            if gpu_id == 0:
                progress_thread = threading.Thread(
                    target=self._monitor_gpu0_progress, 
                    args=(task,),
                    daemon=True
                )
                progress_thread.start()
            
            stdout, stderr = process.communicate()
            
            # Wait for progress monitoring to finish
            if progress_thread and progress_thread.is_alive():
                progress_thread.join(timeout=1.0)
            
            task.end_time = time.time()
            
            # Close progress bar
            if task.progress_bar:
                task.progress_bar.close()
            
            if process.returncode == 0:
                task.status = "completed"
                self._log(f"✅ Completed {task.task_id} on GPU {gpu_id} in {task.duration:.1f}s")
            else:
                task.status = "failed"
                self._log(f"❌ Failed {task.task_id} on GPU {gpu_id}: {stderr[:200]}...")
                
        except Exception as e:
            task.status = "failed"
            task.end_time = time.time()
            if task.progress_bar:
                task.progress_bar.close()
            self._log(f"❌ Exception in {task.task_id} on GPU {gpu_id}: {str(e)}")
        
        return task
    
    def _monitor_gpu0_progress(self, task: TrainingTask):
        """Monitor training progress for GPU 0 tasks by parsing stdout"""
        if not task.progress_bar or not task.process:
            return
        
        # Read stdout line by line to track progress
        while task.process.poll() is None:
            try:
                line = task.process.stdout.readline()
                if line:
                    line = line.strip()
                    
                    # Look for epoch completion indicators
                    if "Epoch" in line and "%" in line:
                        # Extract epoch number from lines like "Epoch 0: 100%|██████████| 1/1 [00:02<00:00,  0.36it/s]"
                        try:
                            # Find epoch number
                            epoch_start = line.find("Epoch ") + 6
                            epoch_end = line.find(":", epoch_start)
                            if epoch_end > epoch_start:
                                epoch_num = int(line[epoch_start:epoch_end])
                                task.progress_bar.n = epoch_num
                                task.progress_bar.refresh()
                        except (ValueError, IndexError):
                            pass
                    
                    # Look for validation loss updates
                    elif "val/total_loss" in line:
                        try:
                            # Extract loss value for display
                            loss_start = line.find("val/total_loss=") + 15
                            loss_end = line.find(",", loss_start)
                            if loss_end == -1:
                                loss_end = line.find("]", loss_start)
                            if loss_end > loss_start:
                                loss_val = float(line[loss_start:loss_end])
                                task.progress_bar.set_postfix({"val_loss": f"{loss_val:.4f}"})
                        except (ValueError, IndexError):
                            pass
                    
                    # Look for training completion
                    elif "Training completed!" in line:
                        task.progress_bar.n = self.args.max_epochs
                        task.progress_bar.refresh()
                        break
                        
            except Exception:
                # If we can't read the line, continue monitoring
                pass
            
            time.sleep(0.1)  # Small delay to avoid busy waiting
    
    def _log(self, message: str):
        """Log message to console and file"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def _get_progress_info(self) -> Dict:
        """Get current progress information"""
        total_tasks = len(self.tasks)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        running = len(self.running_tasks)
        pending = total_tasks - completed - failed - running
        
        return {
            "total": total_tasks,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "progress_percent": (completed + failed) / total_tasks * 100
        }
    
    def _save_results(self):
        """Save training results to file"""
        results = {
            "progress": self._get_progress_info(),
            "completed_tasks": [
                {
                    "task_id": task.task_id,
                    "block_idx": task.block_idx,
                    "linear_type": task.linear_type,
                    "gpu_id": task.gpu_id,
                    "duration": task.duration,
                    "status": task.status
                }
                for task in self.completed_tasks
            ],
            "failed_tasks": [
                {
                    "task_id": task.task_id,
                    "block_idx": task.block_idx,
                    "linear_type": task.linear_type,
                    "gpu_id": task.gpu_id,
                    "status": task.status,
                    "duration": task.duration
                }
                for task in self.failed_tasks
            ]
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def run_parallel_training(self):
        """Main method to run parallel training across all GPUs"""
        self._log("🚀 Starting parallel training across multiple GPUs")
        self._log(f"Total tasks: {len(self.tasks)}")
        
        # Skip already completed tasks
        remaining_tasks = []
        for task in self.tasks:
            if self._check_task_completion(task):
                task.status = "completed"
                self.completed_tasks.append(task)
                self._log(f"⏭️  Skipping {task.task_id} (already completed)")
            else:
                remaining_tasks.append(task)
        
        if not remaining_tasks:
            self._log("🎉 All tasks already completed!")
            return
        
        # Create overall progress bar
        overall_progress = tqdm(
            total=len(remaining_tasks),
            desc="Overall Progress",
            position=1,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Use ThreadPoolExecutor to manage GPU workers
        with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
            
            # Submit initial tasks
            future_to_task = {}
            gpu_assignments = {}  # gpu_id -> future
            
            for gpu_id in range(self.num_gpus):
                if remaining_tasks:
                    task = remaining_tasks.pop(0)
                    future = executor.submit(self._run_task_on_gpu, task, gpu_id)
                    future_to_task[future] = task
                    gpu_assignments[gpu_id] = future
                    self.running_tasks[gpu_id] = task
            
            # Process completed tasks and assign new ones
            while future_to_task:
                # Wait for at least one task to complete
                completed_futures = as_completed(future_to_task)
                
                for future in completed_futures:
                    task = future_to_task.pop(future)
                    
                    try:
                        result_task = future.result()
                        
                        # Remove from running tasks
                        if result_task.gpu_id in self.running_tasks:
                            del self.running_tasks[result_task.gpu_id]
                        
                        # Add to appropriate list
                        if result_task.status == "completed":
                            self.completed_tasks.append(result_task)
                        else:
                            self.failed_tasks.append(result_task)
                        
                        # Update overall progress bar
                        overall_progress.update(1)
                        
                        # Progress update
                        progress = self._get_progress_info()
                        overall_progress.set_postfix({
                            "completed": progress['completed'],
                            "running": progress['running'],
                            "failed": progress['failed']
                        })
                        
                        # Assign new task to this GPU if available
                        if remaining_tasks:
                            new_task = remaining_tasks.pop(0)
                            new_future = executor.submit(self._run_task_on_gpu, new_task, result_task.gpu_id)
                            future_to_task[new_future] = new_task
                            self.running_tasks[result_task.gpu_id] = new_task
                        
                        # Save progress
                        self._save_results()
                        
                    except Exception as e:
                        self._log(f"❌ Exception processing completed task: {str(e)}")
                    
                    break  # Process one completed task at a time
        
        # Close overall progress bar
        overall_progress.close()
        
        # Final summary
        progress = self._get_progress_info()
        self._log(f"🎉 Parallel training completed!")
        self._log(f"✅ Completed: {progress['completed']}")
        self._log(f"❌ Failed: {progress['failed']}")
        self._save_results()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU Task-Parallel Training for Replacement Blocks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and output
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing collected block data (relative to home directory)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for trained models (relative to home directory)")
    
    # Training configuration
    parser.add_argument("--blocks", type=int, nargs="+", required=True,
                       help="Block indices to train (e.g., 1 2 3)")
    parser.add_argument("--linear_types", type=str, nargs="+", 
                       default=["full", "diagonal", "affine"],
                       choices=["full", "diagonal", "affine"],
                       help="Linear layer types to train")
    
    # Training parameters
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension for replacement blocks")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training (increased from 1 due to padding support)")
    parser.add_argument("--max_epochs", type=int, default=50,
                       help="Maximum number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--val_fraction", type=float, default=0.2,
                       help="Fraction of data to use for validation (default: 0.2)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")
    parser.add_argument("--distributed_backend", type=str, default="gloo",
                       choices=["nccl", "gloo", "mpi"],
                       help="Distributed training backend")
    parser.add_argument("--compile", action="store_true", default=False,
                       help="Compile the model using torch.compile")
    
    # Wandb logging arguments
    parser.add_argument("--wandb", action="store_true", default=False,
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="af2distill",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, 
                       default="kryst3154-massachusetts-institute-of-technology",
                       help="Wandb entity (username or team)")
    parser.add_argument("--experiment_name", type=str, default="parallel_block_training",
                       help="Base experiment name for wandb logging")
    
    args = parser.parse_args()
    
    # Run parallel training
    scheduler = GPUTaskScheduler(args)
    scheduler.run_parallel_training()


if __name__ == "__main__":
    main()
