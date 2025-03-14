import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cuml.umap as cuml_umap  # GPU 版 UMAP
from cuml.decomposition import PCA  # GPU 版 PCA
from sklearn.metrics.pairwise import cosine_similarity
import os

# 1. 读取嵌入数据（只计算生成数据的分布）
original_embeddings = np.load("outputs/emb_data/bert/llama_embeddings_original_text.npy")  
generated_response_embeddings = np.load("outputs/emb_data/bert/llama_embeddings_generated_response.npy")  
generated_knowledge_embeddings = np.load("outputs/emb_data/bert/llama_embeddings_generated_knowledge.npy")  

# 2. 计算“生成数据”与“原始数据”的相似性（Cosine Similarity）
print("🔄 Computing Cosine Similarity with Original Data (GPU)...")
response_similarity = cosine_similarity(generated_response_embeddings, original_embeddings)  
knowledge_similarity = cosine_similarity(generated_knowledge_embeddings, original_embeddings)

# 3. 获取最高相似度（取与最相似的原始数据的相似度）
max_response_similarity = np.max(response_similarity, axis=1)
max_knowledge_similarity = np.max(knowledge_similarity, axis=1)

# 4. 计算 Sequence Identity to Training（转换为 0-100%）
sequence_identity = np.concatenate((max_response_similarity, max_knowledge_similarity)) * 100  # 转换为百分比

# 5. **合并“生成数据”嵌入**
generated_embeddings = np.vstack((generated_response_embeddings, generated_knowledge_embeddings))
generated_labels = np.array(["generated_response"] * len(generated_response_embeddings) + 
                            ["generated_knowledge"] * len(generated_knowledge_embeddings))

# 6. **使用 GPU 版 PCA 预降维（4096D → 50D）**
pca_path = "outputs/emb_data/reduced_pca_50_generated.npy"
if os.path.exists(pca_path):
    print("✅ Loading PCA-reduced data from file...")
    pca_50_embeddings = np.load(pca_path)
else:
    print("🔄 Applying GPU PCA (4096 → 50) ...")
    pca = PCA(n_components=50, random_state=42)
    pca_50_embeddings = pca.fit_transform(generated_embeddings)
    np.save(pca_path, pca_50_embeddings)
    print(f"✅ PCA embeddings saved to {pca_path}")

# 7. **使用 GPU 加速 UMAP 降维**
umap_path = "outputs/emb_data/reduced_umap_2d_generated.npy"
if os.path.exists(umap_path):
    print("✅ Loading UMAP-reduced embeddings from file...")
    low_dim_embeddings = np.load(umap_path)
else:
    print("🔄 Applying GPU UMAP on PCA-reduced data ...")
    umap_reducer = cuml_umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.3, random_state=42)
    low_dim_embeddings = umap_reducer.fit_transform(pca_50_embeddings)
    np.save(umap_path, low_dim_embeddings)
    print(f"✅ UMAP embeddings saved to {umap_path}")

# 8. **定义颜色映射（渐变色，基于相似度）**
import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [
    (100/255, 125/255, 125/255),  # 低相似度（深色）
    (160/255, 200/255, 180/255)   # 高相似度（亮色）
])

# 9. **绘制可视化**
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    low_dim_embeddings[:, 0],  
    low_dim_embeddings[:, 1],  
    c=sequence_identity,  # 颜色基于相似度
    cmap=cmap,  
    alpha=0.7,  
    s=30,  
    edgecolors="black"
)

# 10. **添加颜色条（Colorbar）**
cbar = plt.colorbar(sc)
cbar.set_label("% Sequence Identity to Training")

plt.title("Generated Data UMAP (Colored by Similarity to Training Data)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

# **保存图片**
img_path = "outputs/emb_data/embedding_umap_generated_similarity_gpu.png"
plt.savefig(img_path, dpi=300)
print(f"📸 Visualization saved as {img_path}")

# plt.show()
