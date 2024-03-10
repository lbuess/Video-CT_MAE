dependencies: 
	@echo "Initializing Git..."
	git init
	@echo "Installing dependencies..."
	poetry config virtualenvs.in-project true
	poetry install --no-root
	@echo "Adding pre-commit..."
	poetry run pre-commit install

env: dependencies
	@echo "Activating virtual environment..."
	poetry shell