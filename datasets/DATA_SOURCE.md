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
  author = {Haastrup, Simon and Strange, Mikkel and Pandey, Mohnish and Deilmann, Thorsten and Schmidt, Per S. and Hinsche, Nicki F. and Gjerding, Morten N. and Torelli, Daniele and Larsen, Peter M. and Riis-Jensen, Anders C. and Gath, J. and Jacobsen, Karsten W. and Mortensen, Jens J. and Olsen, Thomas and Thygesen, Kristian S.},
  title = {The Computational 2D Materials Database: High-throughput modeling and discovery of atomically thin crystals},
  journal = {2D Materials},
  volume = {5},
  number = {4},
  pages = {042002},
  year = {2018},
  doi = {10.1088/2053-1583/aacfc1}
}


# 2DMatPedia

@article{zhou2019two,
  author = {Zhou, Jun and Shen, Lei and Costa, Miguel Dias and Persson, Kristin A. and Ong, Shyue Ping and Huck, Patrick and Lu, Yunhao and Ma, Xiaoyang and Chen, Yiming and Tang, Hanmei and Feng, Yuan Ping},
  title = {{2DMatPedia}, an open computational database of two-dimensional materials from top-down and bottom-up approaches},
  journal = {Scientific Data},
  volume = {6},
  pages = {86},
  year = {2019},
  doi = {10.1038/s41597-019-0097-3}
}

```

---

## License

Please refer to each database's website for license terms and redistribution policies.
