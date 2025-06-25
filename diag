import streamlit as st
import os
import glob

# ================================
# CODE DE DIAGNOSTIC - DÉBUT
# ================================

st.title("Diagnostic des fichiers disponibles")

# Afficher le répertoire de travail actuel
st.write(f"**Répertoire de travail actuel :** {os.getcwd()}")

# Lister tous les fichiers dans le répertoire racine
st.subheader("Fichiers dans le répertoire racine :")
root_files = os.listdir('.')
for file in sorted(root_files):
    file_path = os.path.join('.', file)
    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path)
        st.write(f"📄 {file} ({file_size} bytes)")
    elif os.path.isdir(file_path):
        st.write(f"📁 {file}/")

# Rechercher récursivement tous les fichiers
st.subheader("Tous les fichiers (récursif) :")
all_files = glob.glob('**/*', recursive=True)
for file_path in sorted(all_files):
    if os.path.isfile(file_path):
        try:
            file_size = os.path.getsize(file_path)
            st.write(f"📄 {file_path} ({file_size} bytes)")
        except:
            st.write(f"📄 {file_path} (taille inaccessible)")
    elif os.path.isdir(file_path):
        st.write(f"📁 {file_path}/")

# Rechercher spécifiquement les fichiers courants
st.subheader("Recherche de fichiers spécifiques :")
extensions_to_check = ['.csv', '.json', '.txt', '.xlsx', '.py', '.md']
for ext in extensions_to_check:
    files_found = glob.glob(f'**/*{ext}', recursive=True)
    if files_found:
        st.write(f"**Fichiers {ext} :**")
        for file in sorted(files_found):
            st.write(f"  - {file}")
    else:
        st.write(f"Aucun fichier {ext} trouvé")

# Vérifier l'existence de fichiers spécifiques (à adapter selon vos besoins)
st.subheader("Vérification de fichiers spécifiques :")
files_to_check = [
    'data.csv',
    'config.json', 
    'requirements.txt',
    'README.md'
]

for file_name in files_to_check:
    if os.path.exists(file_name):
        st.write(f"✅ {file_name} existe")
    else:
        st.write(f"❌ {file_name} n'existe pas")

st.write("---")
st.write("**Fin du diagnostic - Votre application commence ci-dessous**")

# ================================
# CODE DE DIAGNOSTIC - FIN
# ================================

# Votre code d'application commence ici
st.title("Mon Application Streamlit")
# ... reste de votre code ...
