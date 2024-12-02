from matplotlib import pyplot as plt
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import os
import wandb


def plot_strictly_lower_triangular_heatmap(data: list[list], labels: list, save_path: str):
    """
    Plots a heatmap showing only the strictly lower triangular part of the matrix (excluding the diagonal).

    Args:
    - data (list[list]): The 2D data to plot.
    - labels (list[str]): The labels for the x and y axes.
    - title (str): Title of the heatmap.
    - save_path (str): Path to save the heatmap.
    """
    # Convert data to a DataFrame
    df = pd.DataFrame(data, columns=labels, index=labels)
    # Mask the upper triangle including the diagonal
    mask = np.triu(np.ones_like(df, dtype=bool), k=0)
    mask[range(df.shape[0]), range(df.shape[0])] = 0
    print(df)
    # Plot heatmap
    plt.figure(figsize=(4, 4))  # Adjusted for compact size
    sns.heatmap(df, annot=True, fmt=".1f", cmap="Reds", square=True, cbar=False, linewidths=0.5, mask=mask)
    plt.title("Top1 Accuracy", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def createCSV(results: list[list], filename: str, idx2domain:dict):

    with open(filename, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Domain"] + [idx2domain[i] for i in range(len(results))])
        
        for i, result in enumerate(results):
            line = [idx2domain[i]] + [result[j] for j in range(len(result))] + [0] * (len(results) - len(result))
            print(line)
            writer.writerow(line)

def plotStablityPlasticity(plasticity_list: list[list], stability_list: list, save_path: str):
    mean_plasticity = np.mean([domain_result[-1] for domain_result in plasticity_list], axis=0)
    mean_stability = np.mean(stability_list, axis=0)
    # Plot the plasticity and stability as two barplots in the same x
    
    x = np.arange(1)  # Single domain position
    width = 0.35  # Width for each bar
    
    plt.bar(x = len(x), height=mean_plasticity, zorder=1, label="Plasticity")
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, mean_plasticity, width, label="Plasticity", color='orange')
    ax.bar(x + width / 2, mean_stability, width, label="Stability", color='blue')

    # Set x-axis and labels
    ax.set_xticks(x)
    ax.set_ylabel("Value")
    ax.set_title("Plasticity and Stability per Domain")
    ax.legend()

    # Save and close the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_plot_practica(eval_top1_acc, eval_top5_acc, num_domains, idx2domain, prev_accs_top1, prev_accs_top5, save_plot, wandb_exists=True):
    
    avg_task_test_acc = np.mean(eval_top1_acc)
    bar_heights = eval_top1_acc + [0]*(num_domains - len(eval_top1_acc))
    plt.bar(x = range(num_domains), height=bar_heights, zorder=1)
    plt.xticks(range(num_domains), [idx2domain[domain_id] for domain_id in range(num_domains)], rotation='vertical')
    plt.axhline(avg_task_test_acc, c=[0.4]*3, linestyle=':')
    plt.text(0, avg_task_test_acc+0.002, f'(average)', c=[0.4]*3, size=8)
    
    if prev_accs_top1 is not None:
        # plot the previous step's accuracies on top
        # (will show forgetting in red)
        for p, prev_acc_list in enumerate(prev_accs_top1):
            plt.bar(x = range(len(prev_acc_list)), height=prev_acc_list, fc='tab:red', zorder=0, alpha=0.5*((p+1)/len(prev_accs_top1)))
    
    plt.savefig(save_plot + "_top1.png")
    plt.close()

    avg_task_test_acc = np.mean(eval_top5_acc)
    bar_heights = eval_top5_acc + [0]*(num_domains - len(eval_top5_acc))
    plt.bar(x = range(num_domains), height=bar_heights, zorder=1)
    plt.xticks(range(num_domains), [idx2domain[domain_id] for domain_id in range(num_domains)], rotation='vertical')
    plt.axhline(avg_task_test_acc, c=[0.4]*3, linestyle=':')
    plt.text(0, avg_task_test_acc+0.002, f'(average)', c=[0.4]*3, size=8)

    if prev_accs_top5 is not None:
        # plot the previous step's accuracies on top
        # (will show forgetting in red)
        for p, prev_acc_list in enumerate(prev_accs_top5):
            plt.bar(x = range(len(prev_acc_list)), height=prev_acc_list, fc='tab:red', zorder=0, alpha=0.5*((p+1)/len(prev_accs_top5)))
    # Log the plot
    plt.savefig(save_plot + "_top5.png")
    plt.close()
    
    if wandb_exists:
        top1 = plt.imread(save_plot + "_top1.png")
        top5 = plt.imread(save_plot + "_top5.png")
        wandb.log({f'{os.path.basename(save_plot)}' + "_top1": [wandb.Image(top1, caption="Top1 Accuracy")], 
                f'{os.path.basename(save_plot)}' + "_top5": [wandb.Image(top5, caption="Top5 Accuracy")]})
        
#### plotting and accuracy functions:
#
## for this exercise, the training_plot function is able to separate the training curves
## at different timesteps, since these are different tasks with unrelated losses.
## there's also some extra functionality to plot 'soft_loss' from your LwF algorithm, if provided.
#
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#
#import matplotlib as mpl
#
#def training_plot(metrics, task_test_accs, task_test_sets, avg_task_test_acc):      # save the figure to a file
#
#    if show_taskwise_accuracy:
#        bar_heights = task_test_accs + [0]*(len(task_test_sets) - len(selected_test_sets))
#        # display bar plot with accuracy on each evaluation task
#        plt.bar(x = range(len(task_test_sets)), height=bar_heights, zorder=1)
#        plt.xticks(range(len(task_test_sets)), [','.join(task.classes) for task in task_test_sets], rotation='vertical')
#        plt.axhline(avg_task_test_acc, c=[0.4]*3, linestyle=':')
#        plt.text(0, avg_task_test_acc+0.002, f'{model_name} (average)', c=[0.4]*3, size=8)
#
#        if prev_accs is not None:
#            # plot the previous step's accuracies on top
#            # (will show forgetting in red)
#            for p, prev_acc_list in enumerate(prev_accs):
#                plt.bar(x = range(len(prev_acc_list)), height=prev_acc_list, fc='tab:red', zorder=0, alpha=0.5*((p+1)/len(prev_accs)))
#
#        if baseline_taskwise_accs is not None:
#            for t, acc in enumerate(baseline_taskwise_accs):
#                plt.plot([t-0.5, t+0.5], [acc, acc], c='black', linestyle='--')
#
#            # show average as well:
#            baseline_avg = np.mean(baseline_taskwise_accs)
#            plt.axhline(baseline_avg, c=[0.6]*3, linestyle=':')
#            plt.text(0, baseline_avg+0.002, 'baseline average', c=[0.6]*3, size=8)
#
#        plt.show()