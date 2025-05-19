# reset_lr.py
def reset_lr_scheduler(optimizer, current_epoch, reset_interval, initial_lr, final_lr, total_resets):
    # Calculate the learning rate decrement at each reset
    lr_decrement = (initial_lr - final_lr) / total_resets
    
    # Calculate how many resets have occurred
    reset_count = current_epoch // reset_interval
    
    # Calculate the current learning rate after linear decay
    current_lr = initial_lr - (reset_count * lr_decrement)
    current_lr = max(current_lr, final_lr)  # Ensure the learning rate doesn't go below final_lr

    return current_lr
