install:
	pip install -r requirements.txt

build:
	docker build -t sod .

run:
	docker run -dp 8501:8501 sod

local_run:
	streamlit run sod/app.py