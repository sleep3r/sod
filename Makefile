install:
	pip install -r requirements_local.txt

build:
	docker build -t sod .

run:
	docker run -p $(PORT):8501 sod

local_run:
	streamlit run sod/app.py