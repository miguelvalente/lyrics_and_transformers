# Lyrics And BERT

This is an ongoing project that makes use of my own lyrics dataset created using my <a href="https://github.com/miguelvalente/lyrics-scraper" target="_blank">`scraper`</a>. Currently I'm trainning several models, and after I'll do some form of analysis on the results. You can see the results on my <a href="https://app.wandb.ai/mvalente/lyrics-classifier" target="_blank">`wandb profile`</a>

---


### Arguments

All arguments have a default but you can specify the following paramenters:

- --lyrics_path
- --batch_size
- --epochs
- --wandb_project_name
You can choose between BERT and DistilBERT based on a numerical value 0 = BERT, 1 = DistilBert
- --model_choice

### TO DO
- [x] Lyrics Genre Classification
- [ ] Lyrics Generation
- [x] Lyrics Genre Classification - Analysis
- [ ] Lyrics Generation - Analysis
