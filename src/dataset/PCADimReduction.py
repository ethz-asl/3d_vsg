import torch
from sklearn.decomposition import PCA


class PCADimReduction:
    """ Applies Principal Component Analysis to embedding matrix. """
    def __call__(self, embedding_tensor: torch.Tensor, num_dims=480) -> torch.Tensor:
        np_input_mat = embedding_tensor.numpy()
        pca = PCA(n_components=num_dims)
        pca.fit(np_input_mat)
        n_samples = embedding_tensor.shape[0]
        reduced_embeddings = torch.zeros((n_samples, num_dims))
        for i in range(n_samples):
            reduced_embeddings[i, :] = torch.Tensor(pca.transform(np_input_mat[i, :].reshape(1, -1)))

        return reduced_embeddings

