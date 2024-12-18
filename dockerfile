# Utiliser une image de base avec Python 3.9
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances à partir de requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code du projet dans le conteneur
COPY . .

# Exécuter le script principal
CMD ["python", "main.py"]
