# ZPJa: Summarization - Methods Comparison

This projects focuses on comparison of several summarization techniques. See more in
the [documentation](documentation.pdf). Scripts used for examination of methods are present in the `scripts` directory.

## Prerequisites
1. Load environment and activate it.
```bash
conda env create -f environment.yml
conda env activate zpja_summarization
```

2. Create `data` directory.
```bash
mkdir data
```

3. Download models and datasets.
   1. Glove word vectors - https://www.kaggle.com/danielwillgeorge/glove6b100dtxt (Save as `data/glove.6B.100d.txt` )
   2. Wikihow dataset - https://ucsb.box.com/s/ap23l8gafpezf4tq3wapr6u8241zz358 (Save as `data/wikihowAll.csv`)
   3. Distilbert checkpoint - https://www.googleapis.com/drive/v3/files/1WxU7cHECfYaU32oTM0JByTRGS5f6SYEF?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE (Save as `data/distilbert_ext.pt`)

