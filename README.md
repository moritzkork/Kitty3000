# Kitty3000 – front end

Streamlit demo for the Kitty3000 API (cat sound → mood). Separate repo so Streamlit Community Cloud only installs what the front end needs, not torch.

## Run locally

    uv sync
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    uv run streamlit run app/Home.py

`secrets.toml` holds the API URL and is gitignored. The example file has the Cloud Run URL.

## Deploy

Streamlit Community Cloud, repo `moritzkork/Kitty3000`, main file `app/Home.py`, Python 3.12. App settings → Secrets → the same `API_URL` line as in `secrets.toml.example`. Every push to the deployed branch redeploys.

## Samples

| file | source |
|---|---|
| cat_complaining.mp3 | BigSoundBank.com ID 0658 |
| cat_meow.mp3 | BigSoundBank.com ID 1890 |
| two_cats_fighting.mp3 | BigSoundBank.com ID 0817 |
| human_imitating_a_cat.mp3 | BigSoundBank.com ID 0926 |
| resting_training_clip.mp3 | from the training dataset – the models have seen this one |

BigSoundBank clips (Joseph Sardin, free to use) are cut to 15 s around the loudest part and re-encoded, the originals have broken ID3 headers that libsndfile refuses.
