# image-alignment

Image alignment using openCV.


### Usage

```
python main.py --ref tis1.jpg --algn tis2.jpg --out align_tis.jpg --matches 20000 --top 0.5 --debug True
python main.py --ref vit2.jpg --algn vit1.jpg --out align_vit.jpg --matches 200 --top 0.15              
```

### Args

```
--ref : reference image path

--algn : shifted image path

--out : saved aligned image name

--matches : number of matches to be observed (default: 200)

--top : top matches (in percentage) that we want to consider (default: 0.15)

--debug : set to True to see matching process

```

