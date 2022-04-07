# KoBART-dialect

## Data preparing

- First, download the file below from [aihub](https://aihub.or.kr/aihub-data/natural-language/about) and set it up as follows.

```
.
└── data/
│   ├── 한국어 방언 발화 데이터(강원도)
│   │   ├── Training/[라벨]강원도_학습데이터_1.zip
│   │   └── Validation/[라벨]강원도_학습데이터_2.zip
│   ├── 한국어 방언 발화 데이터(경상도)
│   │   ├── Training/[라벨]경상도_학습데이터_1.zip
│   │   └── Validation/[라벨]경상도_학습데이터_2.zip
│   ├── 한국어 방언 발화 데이터(전라도)
│   │   ├── Training/[라벨]전라도_학습데이터_1.zip
│   │   └── Validation/[라벨]전라도_학습데이터_2.zip
│   ├── 한국어 방언 발화 데이터(제주도)
│   │   ├── Training/[라벨]제주도_학습데이터_1.zip
│   │   └── Validation/[라벨]제주도_학습데이터_3.zip
│   └── 한국어 방언 발화 데이터(충청도)
│       ├── Training/[라벨]충청도_학습데이터_1.zip
│       └── Validation/[라벨]충청도_학습데이터_2.zip
├── kodialect/..
├── .gitignore
├── LICENSE
└── README.md
```

- Second, unzip files

```shell
$ sh unzip.sh
```

- Third, run `prepare_data.py`
    - There may be errors in the json data itself provided by aihub. Please refer to the [issue](https://github.com/jinmang2/KoBART-dialect/issues/1) and edit the file directly and run the above python script.

```shell
$ python prepare_data.py
```

- Final data folder

```
.
└── data/
│   ├── chungcheongdo/..
│   ├── gangwondo/..
│   ├── gyeongsangdo/..
│   ├── jejudo/..
│   ├── jeollado/..
│   ├── style_classification/..
│   ├── style_transfer/..
│   ├── train_dialect.json
│   └── valid_dialect.json
├── kodialect/..
├── .gitignore
├── LICENSE
└── README.md
```


## Citations

```
@inproceedings{lai-etal-2021-thank,
    title = "Thank you {BART}! Rewarding Pre-Trained Models Improves Formality Style Transfer",
    author = "Lai, Huiyuan and Toral, Antonio and Nissim, Malvina",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.62",
    doi = "10.18653/v1/2021.acl-short.62",
    pages = "484--494",
}
```
