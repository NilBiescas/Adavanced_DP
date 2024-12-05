from sklearn.metrics.pairwise import cosine_similarity
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

def compute_color_histograms(image_paths, root_images, num_bins=256, sample_size=1000):
    histograms = []
    if sample_size and len(image_paths) > sample_size:
        image_paths = np.random.choice(image_paths, sample_size, replace=False)

    print(f"Processing {len(image_paths)} images for histogram computation...")

    last_log_time = time.time()
    for idx, img_rel_path in enumerate(image_paths):
        img_abs_path = os.path.join(root_images, img_rel_path)
        if not os.path.exists(img_abs_path):
            print(f"Missing file: {img_abs_path}")
            continue  # Skip missing files

        img = cv2.imread(img_abs_path)
        if img is None:
            print(f"Unreadable file: {img_abs_path}")
            continue  # Skip unreadable files

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Flatten the image
        img_flat = img.reshape(-1, 3)
        # Compute histogram for each channel
        hist_r, _ = np.histogram(img_flat[:, 0], bins=num_bins, range=(0, 256), density=True)
        hist_g, _ = np.histogram(img_flat[:, 1], bins=num_bins, range=(0, 256), density=True)
        hist_b, _ = np.histogram(img_flat[:, 2], bins=num_bins, range=(0, 256), density=True)
        # Concatenate histograms
        hist = np.concatenate([hist_r, hist_g, hist_b])
        histograms.append(hist)

        # Log progress periodically
        if time.time() - last_log_time > 30:  # Log every 30 seconds
            print(f"Processed {idx + 1}/{len(image_paths)} images...")
            last_log_time = time.time()

    return np.array(histograms)

def compute_cosine_similarities(domain_histograms, domain_names):
    """
    Compute pairwise domain similarities using Cosine Similarity.

    Args:
        domain_histograms (dict): Dictionary of domain histograms.
        domain_names (list): List of domain names.

    Returns:
        similarity_matrix (np.ndarray): Matrix of pairwise similarities.
    """
    num_domains = len(domain_names)
    similarity_matrix = np.zeros((num_domains, num_domains))

    for i in range(num_domains):
        for j in range(num_domains):
            if i <= j:  # Compute only upper triangle and mirror
                hists_i = domain_histograms[domain_names[i]]
                hists_j = domain_histograms[domain_names[j]]
                # Compute cosine similarity between histograms of the two domains
                cos_sim = cosine_similarity(hists_i, hists_j)
                avg_sim = np.mean(cos_sim)  # Average over all pairwise comparisons
                similarity_matrix[i, j] = avg_sim
                similarity_matrix[j, i] = avg_sim  # Symmetric matrix
        print(f"Computed similarities for domain {domain_names[i]}.")

    return similarity_matrix

def save_similarity_matrix_plot(matrix, domains, save_path="domain_cosine_similarity_matrix.png"):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, interpolation='nearest', cmap='viridis')
    plt.title('Domain Similarity Matrix (Cosine Similarity)')
    plt.colorbar(label='Cosine Similarity')
    plt.xticks(range(len(domains)), domains, rotation=45)
    plt.yticks(range(len(domains)), domains)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Similarity matrix saved as {save_path}")

def main():
    import random
    random.seed(42)
    np.random.seed(42)
    
    root_dn4il = '/fhome/amlai07/Adavanced_DP/Data/DN4IL'      # Path to domain text files
    root_images = '/fhome/amlai07/Adavanced_DP/Data/domainnet'  # Root directory of images
    domains = ['real', 'painting']

    domain_histograms = {}
    num_bins = 64  # Number of bins for histograms
    sample_size = 500  # Number of images to sample per domain

    print("Starting domain histogram computation...")

    for domain in domains:
        domain_file = os.path.join(root_dn4il, f'{domain}_train.txt')
        if not os.path.exists(domain_file):
            print(f"Domain file not found: {domain_file}")
            continue

        print(f"\nProcessing domain: {domain}")
        with open(domain_file, 'r') as f:
            lines = f.readlines()

        image_paths = [line.strip().split()[0] for line in lines]

        # Compute histograms for the domain
        histograms = compute_color_histograms(image_paths, root_images, num_bins=num_bins, sample_size=sample_size)
        domain_histograms[domain] = histograms

    print("\nCompleted histogram computation for all domains.")
    print("Starting domain similarity computation...")

    # Compute domain similarities using cosine similarity
    similarity_matrix = compute_cosine_similarities(domain_histograms, domains)

    print("\nDomain Similarity Matrix (Cosine Similarity):")
    print("Domains:", domains)
    print(similarity_matrix)

    # Save the similarity matrix plot
    save_similarity_matrix_plot(similarity_matrix, domains, save_path="domain_cosine_similarity_matrix2.png")

if __name__ == "__main__":
    main()
