# Data Sources

This project uses two established 2D materials databases for training and evaluation.

---

## C2DB (Computational 2D Materials Database)

| Property | Value |
|----------|-------|
| **Name** | C2DB |
| **Source** | Computational 2D Materials Database |
| **URL** | https://www.2dhub.org/c2db/c2db.html |
| **Institution** | Technical University of Denmark (DTU) |

### Download

```bash
# Download C2DB dataset
curl -O https://www.2dhub.org/c2db/c2db.tar.gz
tar -xzf c2db.tar.gz
```

---

## 2DMatPedia

| Property | Value |
|----------|-------|
| **Name** | 2DMatPedia |
| **Source** | 2DMatPedia Database |
| **URL** | http://www.2dmatpedia.org/ |
| **Institution** | Nanyang Technological University |

### Download

```bash
# Download 2DMatPedia dataset
curl -O http://www.2dmatpedia.org/download
```

---

## Citation

If you use these datasets, please cite the original sources:

```bibtex
# C2DB
@article{haastrup2018computational,
  title={Computational 2D materials database: electronic structure of CRUD and MXenes},
  author={Haastrup, Simon and Strange, Mikkel and Pandey, Mukhtar and others},
  journal={2D Materials},
  volume={5},
  pages={042002},
  year={2018}
}

# 2DMatPedia
@article{zhou2021two,
  title={Two-dimensional inorganic piezoelectric materials for mechanical energy harvesting},
  author={Zhou, Yao and others},
  journal={Nature Communications},
  year={2021}
}
```

---

## License

Please refer to each database's website for license terms and redistribution policies.
