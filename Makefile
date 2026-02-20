.PHONY: data features train evaluate app validate clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

data: ## Download, validate, and process data
	python -m data.processed.download
	python -m data.validate_data
	python -m data.processed.process_data

features: ## Build feature matrix from processed data
	python -m features.build_features

train: ## Train the model
	python train.py

evaluate: ## Evaluate the model (runs train.py which includes evaluation)
	python train.py

app: ## Launch the Streamlit app
	streamlit run app.py

validate: ## Run data validation only
	python -m data.validate_data

clean: ## Remove generated data, features, and model artifacts
	rm -rf data/raw/*.parquet
	rm -rf data/processed/*.parquet
	rm -rf data/plots/*.png
	rm -rf features/feature_matrix.parquet
	rm -rf models/*.pkl models/*.joblib
	rm -rf models/saved/*.joblib
	rm -rf mlruns/
