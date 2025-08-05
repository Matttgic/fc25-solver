import zipfile
import os
import subprocess

# Étape 1 : Décompresser le .zip s'il n'est pas déjà extrait
ZIP_FILE = "fc25-solver.zip"
EXTRACTED_FILE = "player-data-full-2025-june.csv"

if os.path.exists(ZIP_FILE) and not os.path.exists(EXTRACTED_FILE):
    print("Décompression de l'archive...")
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("✅ Décompression terminée.")

# Étape 2 : Lancer le vrai app.py maintenant décompressé
if os.path.exists("app.py") and __file__ != os.path.abspath("app.py"):
    print("⏳ Lancement de la vraie application...")
    subprocess.run(["python", "app.py"])
else:
    print("❌ app.py non trouvé après décompression.")
