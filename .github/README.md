## MLX SSL

Python package for self-supervised learning on Apple Silicon using MLX

### Model Implementations

<details><summary>Data2Vec</summary>

```python
from mlx_ssl.models import Data2VecVisionForImageClassification


model = Data2VecVisionForImageClassification.from_pretrained(
    "mlx-community/data2vec-vision-large-ft1k"
)
```

* https://arxiv.org/abs/2202.03555
```

</details>