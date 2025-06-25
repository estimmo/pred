import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Prédiction Valeur Foncière",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #e8f4fd;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏠 Prédiction de Valeur Foncière d une Maison</h1>', unsafe_allow_html=True)
st.markdown("---")

# Enhanced function to load geographical data
@st.cache_data
def load_geo_data():
    """Load geographical data from CSV file with enhanced error handling"""
    file_path = 'Liste_Departement_Ville_ValeursFoncieres-all_fitre.csv'
    
    try:
        # First, try to detect the file structure
        sample = pd.read_csv(file_path, nrows=5)
        
        # Check if first row contains headers
        has_headers = not all(sample.iloc[0].apply(lambda x: str(x).isdigit()))
        
        # Load the complete file
        if has_headers:
            geo_df = pd.read_csv(file_path, header=0)
            if len(geo_df.columns) >= 3:
                geo_df.columns = ['Code_departement', 'Code_postal', 'Commune']
        else:
            geo_df = pd.read_csv(file_path, 
                               names=['Code_departement', 'Code_postal', 'Commune'],
                               header=None)
        
        # Data cleaning and type conversion
        geo_df['Code_departement'] = pd.to_numeric(geo_df['Code_departement'], errors='coerce')
        geo_df['Code_postal'] = pd.to_numeric(geo_df['Code_postal'], errors='coerce')
        
        # Remove rows with missing values
        initial_count = len(geo_df)
        geo_df = geo_df.dropna(subset=['Code_departement', 'Code_postal'])
        cleaned_count = len(geo_df)
        
        # Convert to appropriate types
        geo_df['Code_departement'] = geo_df['Code_departement'].astype('int32')
        geo_df['Code_postal'] = geo_df['Code_postal'].astype('int32')
        geo_df['Commune'] = geo_df['Commune'].str.strip()
        
        # Sort for better user experience
        geo_df = geo_df.sort_values(['Code_departement', 'Code_postal', 'Commune'])
        
        logger.info(f"Geographical data loaded: {cleaned_count}/{initial_count} rows")
#        st.success(f"✅ Données géographiques chargées: {cleaned_count:,} lignes")
        
        if cleaned_count < initial_count:
            st.info(f"ℹ️ {initial_count - cleaned_count} lignes avec des données manquantes ont été supprimées")
        
        return geo_df
        
    except FileNotFoundError:
        st.error(f"❌ Fichier '{file_path}' introuvable")
        st.info("Assurez-vous que le fichier CSV est dans le répertoire de l'application")
        return pd.DataFrame(columns=['Code_departement', 'Code_postal', 'Commune'])
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        with st.expander("🔧 Format de fichier attendu"):
            st.code("Code_departement,Code_postal,Commune\n78,78980,SAINT-ILLIERS-LE-BOIS")
        return pd.DataFrame(columns=['Code_departement', 'Code_postal', 'Commune'])

# Enhanced function to load model and encoders
@st.cache_resource
def load_model_and_encoders(model_choice):
    """Load model and encoders with enhanced error handling"""
    
    # CORRECTION: Essayer les deux noms de fichiers possibles pour CatBoost
    if model_choice == "cb":
        # Essayer d'abord model_cb_b.pkl, puis model_cb.pkl
        possible_paths = ['model_cb_b.pkl', 'model_cb.pkl']
        model_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                st.info(f"📁 Fichier CatBoost trouvé: {path}")
                break
        
        if model_path is None:
            st.error(f"❌ Aucun fichier CatBoost trouvé. Fichiers recherchés: {possible_paths}")
            return None, None, {}
            
    elif model_choice == "lgb":
        model_path = 'model_lgb.pkl'
    else:
        model_path = f'model_{model_choice}.pkl'
    
    encoders_path = 'encoders.pkl'
    
    model = None
    encoders = None
    model_info = {}
    
    # Load model with multiple protocols and diagnostics
    try:
        if os.path.exists(model_path):
            # First, let's diagnose the file
            file_size = os.path.getsize(model_path)
            st.info(f"📁 Fichier utilisé: {model_path} ({file_size} bytes)")
            
            # Check if file is empty or too small
            if file_size < 10:
                st.error("❌ Le fichier est trop petit ou vide")
                return None, None, {}
            
            # Read first few bytes to check file signature
            try:
                with open(model_path, 'rb') as f:
                    first_bytes = f.read(20)
                    st.info(f"🔍 Premiers bytes du fichier: {first_bytes[:10]}")
                    
                    # Check for Git LFS pointer file
                    if first_bytes.startswith(b'version ht'):
                        st.error("🚨 FICHIER GIT LFS DÉTECTÉ!")
                        st.error("Le fichier est un pointeur Git LFS, pas le modèle réel.")
                        
                        # Show file content to confirm
                        with open(model_path, 'r', encoding='utf-8') as f:
                            lfs_content = f.read()
                            st.code(lfs_content, language='text')
                        
                        st.info("💡 **Solutions pour récupérer le vrai fichier modèle:**")
                        st.write("**Option 1 - Git LFS:**")
                        st.code("git lfs pull", language='bash')
                        st.write("**Option 2 - Téléchargement manuel:**")
                        st.write("1. Allez sur votre repository Git")
                        st.write("2. Cliquez sur le fichier .pkl")
                        st.write("3. Cliquez sur 'Download' pour obtenir le vrai fichier")
                        st.write("**Option 3 - Re-génération:**")
                        st.write("Re-entraînez le modèle et sauvegardez-le directement")
                        
                        return None, None, {}
                    
                    # Check if it looks like a pickle file
                    elif not (first_bytes.startswith(b'\x80') or first_bytes.startswith(b'(') or first_bytes.startswith(b']')):
                        st.warning("⚠️ Le fichier ne semble pas être un fichier pickle standard")
            except Exception as e:
                st.warning(f"⚠️ Impossible de lire les premiers bytes: {e}")
            
            # Try different pickle protocols and encoding methods
            model_loaded = False
            
            # Method 1: Default pickle load
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                model_loaded = True
                st.success(f"✅ Modèle {model_choice.upper()} chargé avec succès (méthode standard)!")
            except Exception as e1:
                st.warning(f"⚠️ Méthode standard échouée: {str(e1)}")
                
                # Method 2: Try with different pickle protocol
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f, encoding='latin1')
                    model_loaded = True
                    st.success(f"✅ Modèle {model_choice.upper()} chargé avec succès (encoding latin1)!")
                except Exception as e2:
                    st.warning(f"⚠️ Méthode latin1 échouée: {str(e2)}")
                    
                    # Method 3: Try with bytes encoding
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f, encoding='bytes')
                        model_loaded = True
                        st.success(f"✅ Modèle {model_choice.upper()} chargé avec succès (encoding bytes)!")
                    except Exception as e3:
                        st.warning(f"⚠️ Méthode bytes échouée: {str(e3)}")
                        
                        # Method 4: Try joblib (common for scikit-learn models)
                        try:
                            import joblib
                            model = joblib.load(model_path)
                            model_loaded = True
                            st.success(f"✅ Modèle {model_choice.upper()} chargé avec succès (joblib)!")
                        except Exception as e4:
                            st.warning(f"⚠️ Méthode joblib échouée: {str(e4)}")
                            
                            # Method 5: Try opening as text to see content
                            try:
                                with open(model_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content_preview = f.read(100)
                                    st.error("🔍 Aperçu du contenu du fichier (premiers 100 caractères):")
                                    st.code(content_preview)
                            except:
                                pass
                            
                            st.error(f"❌ Toutes les méthodes de chargement ont échoué:")
                            st.error(f"  - Standard: {str(e1)}")
                            st.error(f"  - Latin1: {str(e2)}")
                            st.error(f"  - Bytes: {str(e3)}")
                            st.error(f"  - Joblib: {str(e4)}")
                            
                            # Additional diagnostic info
                            st.info("💡 Suggestions de résolution:")
                            st.write("1. Le fichier semble corrompu ou dans un format non reconnu")
                            st.write("2. Vérifiez que le fichier a été correctement transféré")
                            st.write("3. Re-générez le fichier .pkl depuis votre environnement d'entraînement")
                            st.write("4. Le fichier pourrait être dans un format différent (joblib, dill, etc.)")
                            
                            return None, None, {}
            
            if model_loaded:
                logger.info(f"{model_choice.upper()} model loaded successfully")
                
                # NOUVEAU: Extraire les informations du modèle pour diagnostic
                try:
                    if hasattr(model, 'feature_names_in_'):
                        model_info['expected_features'] = list(model.feature_names_in_)
                        st.info(f"🔍 Modèle attend {len(model.feature_names_in_)} features")
                    if hasattr(model, 'n_features_in_'):
                        model_info['n_features'] = model.n_features_in_
                except:
                    pass
                
        else:
            st.error(f"❌ Fichier '{model_path}' introuvable")
            return None, None, {}
    except Exception as e:
        st.error(f"❌ Erreur inattendue lors du chargement du modèle: {str(e)}")
        return None, None, {}
    
    # Load encoders with similar error handling
    try:
        if os.path.exists(encoders_path):
            try:
                with open(encoders_path, 'rb') as f:
                    encoders = pickle.load(f)
                st.success("✅ Encodeurs chargés avec succès!")
            except:
                # Try with encoding fallback for encoders too
                try:
                    with open(encoders_path, 'rb') as f:
                        encoders = pickle.load(f, encoding='latin1')
                    st.success("✅ Encodeurs chargés avec succès (encoding latin1)!")
                except:
                    st.warning("⚠️ Erreur lors du chargement des encodeurs - utilisation par défaut")
                    encoders = None
            
            logger.info("Encoders loaded successfully")
            
            # NOUVEAU: Afficher les clés d'encodage disponibles
            if isinstance(encoders, dict):
                st.info(f"🔑 Encodeurs disponibles: {list(encoders.keys())}")
    except Exception as e:
        st.warning(f"⚠️ Erreur lors du chargement des encodeurs: {str(e)}")
        st.info("Utilisation de l'encodage par défaut")
    
    return model, encoders, model_info


def encode_categorical_features(input_data, encoders=None):
    """Encode categorical features with EXACT Jupyter consistency"""
    input_encoded = input_data.copy()
    original_values = {}
    encoding_log = []  # NOUVEAU: Log des encodages pour diagnostic
    
    try:
        if encoders is not None and isinstance(encoders, dict):
            # CORRECTION: Encodage strict selon les encodeurs sauvegardés
            for col in ['Nature_mutation', 'Type_local', 'Commune']:
                if col in input_encoded.columns:
                    original_val = input_encoded[col].iloc[0]
                    original_values[col] = original_val
                    
                    if col in encoders:
                        try:
                            # CRITIQUE: Vérifier si la valeur existe dans l'encodeur
                            if hasattr(encoders[col], 'classes_'):
                                # LabelEncoder
                                if original_val in encoders[col].classes_:
                                    encoded_val = encoders[col].transform([original_val])[0]
                                    encoding_log.append(f"{col}: '{original_val}' -> {encoded_val}")
                                else:
                                    # Valeur inconnue - utiliser la stratégie par défaut
                                    encoded_val = -1  # Ou une autre stratégie
                                    encoding_log.append(f"{col}: '{original_val}' -> {encoded_val} (inconnu)")
                            else:
                                # Autre type d'encodeur
                                encoded_val = encoders[col].transform([original_val])[0]
                                encoding_log.append(f"{col}: '{original_val}' -> {encoded_val}")
                            
                            input_encoded[col] = encoded_val
                            
                        except Exception as e:
                            # Fallback si l'encodage échoue
                            if col == 'Nature_mutation':
                                encoded_val = 0  # Vente = 0
                            elif col == 'Type_local':
                                encoded_val = 0  # Maison = 0  
                            elif col == 'Commune':
                                encoded_val = abs(hash(str(original_val))) % 10000
                            
                            input_encoded[col] = encoded_val
                            encoding_log.append(f"{col}: '{original_val}' -> {encoded_val} (fallback)")
                            logger.warning(f"Fallback encoding for {col}: {e}")
                    else:
                        # Encodeur non disponible - utiliser valeurs par défaut
                        if col == 'Nature_mutation':
                            encoded_val = 0
                        elif col == 'Type_local':
                            encoded_val = 0
                        elif col == 'Commune':
                            encoded_val = abs(hash(str(original_val))) % 10000
                        
                        input_encoded[col] = encoded_val
                        encoding_log.append(f"{col}: '{original_val}' -> {encoded_val} (défaut)")
        else:
            # Pas d'encodeurs - utiliser l'encodage par défaut
            for col in ['Nature_mutation', 'Type_local', 'Commune']:
                if col in input_encoded.columns:
                    original_val = input_encoded[col].iloc[0]
                    original_values[col] = original_val
                    
                    if col == 'Nature_mutation':
                        encoded_val = 0  # Vente = 0
                    elif col == 'Type_local':
                        encoded_val = 0  # Maison = 0
                    elif col == 'Commune':
                        encoded_val = abs(hash(str(original_val))) % 10000
                    
                    input_encoded[col] = encoded_val
                    encoding_log.append(f"{col}: '{original_val}' -> {encoded_val} (défaut)")
            
    except Exception as e:
        st.error(f"Erreur lors de l'encodage: {str(e)}")
        raise e
    
    return input_encoded, original_values, encoding_log

def prepare_input_data_exact(code_departement, code_postal, commune, surface_reelle_bati, 
                           nombre_pieces_principales, surface_terrain, moyenne_taux):
    ligne = pd.DataFrame([{
        "Nature_mutation": "Vente",
        "Code_postal": code_postal,
        "Commune": commune,
        "Code_departement": code_departement,
        "Type_local": "Maison",
        "Surface_reelle_bati": surface_reelle_bati,
        "Nombre_pieces_principales": nombre_pieces_principales,
        "Surface_terrain": surface_terrain,
        "Moyenne_Taux": moyenne_taux
    }])
    
    # MÊME conversion qu'app.py
    categorical_cols = ["Nature_mutation", "Commune", "Type_local"]
    for col in categorical_cols:
        ligne[col] = ligne[col].astype("category")
    
    return ligne

def validate_inputs(**kwargs):
    """Validate user inputs"""
    errors = []
    
    if kwargs['surface_reelle_bati'] <= 0:
        errors.append("La surface bâtie doit être positive")
    
    if kwargs['nombre_pieces_principales'] <= 0:
        errors.append("Le nombre de pièces doit être positif")
    
    if kwargs['surface_terrain'] < 0:
        errors.append("La surface du terrain ne peut pas être négative")
    
    if kwargs['moyenne_taux'] < 0:
        errors.append("Le taux d'intérêt ne peut pas être négatif")
    
    return errors

def make_prediction_exact(model, input_data, encoders=None):
    """Make prediction with EXACT Jupyter methodology"""
    try:
        # ÉTAPE 1: Encodage exact
        input_encoded, original_values, encoding_log = encode_categorical_features(input_data, encoders)
        
        # ÉTAPE 2: Vérification de l'ordre des colonnes
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            current_features = list(input_encoded.columns)
            
            # CRITIQUE: Réorganiser les colonnes dans l'ordre attendu
            if set(expected_features) == set(current_features):
                input_encoded = input_encoded[expected_features]
                column_order_msg = "✅ Ordre des colonnes ajusté selon le modèle"
            else:
                missing_features = set(expected_features) - set(current_features)
                extra_features = set(current_features) - set(expected_features)
                column_order_msg = f"⚠️ Différences features: manquantes={missing_features}, extra={extra_features}"
        else:
            column_order_msg = "ℹ️ Ordre des colonnes non vérifié (pas d'info modèle)"
        
        # ÉTAPE 3: Prédiction
        prediction = model.predict(input_encoded)[0]
        
        return {
            'prediction': prediction,
            'method': "Prédiction avec encodage exact",
            'input_encoded': input_encoded,
            'original_values': original_values,
            'encoding_log': encoding_log,
            'column_order_msg': column_order_msg,
            'success': True
        }
    
    except Exception as e:
        return {
            'prediction': None,
            'method': f"Erreur: {str(e)}",
            'input_encoded': None,
            'original_values': {},
            'encoding_log': [],
            'column_order_msg': "",
            'success': False,
            'error': str(e)
        }

# Load geographical data
geo_df = load_geo_data()

# Add model selection in sidebar
st.sidebar.subheader("🤖 Sélection du Modèle")
model_choice = st.sidebar.selectbox(
    "Choisissez le modèle de prédiction:",
    options=["lgb", "cb"],
    format_func=lambda x: f"Model {x.upper()}" + (" (LightGBM)" if x == "lgb" else " (CatBoost)"),
    help="Sélectionnez le modèle à utiliser pour la prédiction"
)

# Load model and encoders based on user choice
model, encoders, model_info = load_model_and_encoders(model_choice)

# NOUVEAU: Section de diagnostic du modèle
#if model is not None and model_info:
 #   with st.expander("🔍 Diagnostic du Modèle (pour débogage)"):
  #      if 'expected_features' in model_info:
   #         st.write("**Features attendues par le modèle:**")
    #        for i, feature in enumerate(model_info['expected_features']):
     #           st.write(f"{i+1}. {feature}")
      #  
  #      if 'n_features' in model_info:
   #         st.write(f"**Nombre de features:** {model_info['n_features']}")
    #    
     #   if encoders:
      #      st.write("**Encodeurs disponibles:**")
       #     for key, encoder in encoders.items():
        #        if hasattr(encoder, 'classes_'):
         #           st.write(f"- {key}: {len(encoder.classes_)} classes")
          #      else:
           #         st.write(f"- {key}: {type(encoder).__name__}")

# Main application logic
if not geo_df.empty and model is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🔧 Paramètres de prédiction")
        
        # Fixed parameters
        with st.container():
            st.markdown("**Paramètres fixes:**")
            col_fixed1, col_fixed2 = st.columns(2)
            with col_fixed1:
                st.text_input("Nature mutation", value="Vente", disabled=True)
            with col_fixed2:
                st.text_input("Type local", value="Maison", disabled=True)
        
        st.markdown("---")
        st.subheader("📍 Localisation")
        
        # Location selection with improved UX
        departements_disponibles = sorted(geo_df['Code_departement'].unique())
        code_departement = st.selectbox(
            "Numéro du Département",
            options=departements_disponibles,
            help="Sélectionnez le département"
        )
        
        if code_departement:
            codes_postaux_filtres = sorted(
                geo_df[geo_df['Code_departement'] == code_departement]['Code_postal'].unique()
            )
            code_postal = st.selectbox(
                "Code Postal",
                options=codes_postaux_filtres,
                help=f"{len(codes_postaux_filtres)} codes postaux disponibles"
            )
        else:
            code_postal = None
            st.selectbox("Code Postal", options=[], disabled=True, 
                        help="Sélectionnez d'abord un département")
        
        if code_postal:
            communes_filtrees = sorted(
                geo_df[
                    (geo_df['Code_departement'] == code_departement) & 
                    (geo_df['Code_postal'] == code_postal)
                ]['Commune'].unique()
            )
            commune = st.selectbox(
                "Commune",
                options=communes_filtrees,
                help=f"{len(communes_filtrees)} commune(s) disponible(s)"
            )
        else:
            commune = None
            st.selectbox("Commune", options=[], disabled=True,
                        help="Sélectionnez d'abord un code postal")
        
        st.markdown("---")
        st.subheader("🏘️ Caractéristiques du bien")
        
        # Property characteristics with better validation
        surface_reelle_bati = st.number_input(
            "Surface habitable (m²)",
            min_value=40.0,
            max_value=2000.0,
            value=90.0,
            step=1.0,
            help="Surface habitable du logement"
        )
        
        nombre_pieces_principales = st.number_input(
            "Nombre de pièces",
            min_value=1,
            max_value=20,
            value=4,
            step=1,
            help="Chambres, salon, salle à manger..."
        )
        
        surface_terrain = st.number_input(
            "Surface total du terrain (m²)",
            min_value=0.0,
            max_value=100000.0,
            value=200.0,
            step=10.0,
            help="Surface totale du terrain"
        )
        
        moyenne_taux = st.number_input(
            "Taux d'intérêt moyen d'emprunt (%)",
            min_value=0.0,
            max_value=15.0,
            value=3.5,
            step=0.1,
            help="Taux d'intérêt immobilier actuel"
        )
        

    
    with col2:
        st.subheader("🔮 Résultat de la prédiction")
        
        # Show selected model
        st.info(f"🤖 Modèle sélectionné: **{model_choice.upper()}**")
        
        # Validation
        validation_errors = validate_inputs(
            surface_reelle_bati=surface_reelle_bati,
            nombre_pieces_principales=nombre_pieces_principales,
            surface_terrain=surface_terrain,
            moyenne_taux=moyenne_taux
        )
        
        tous_champs_remplis = all([
            code_departement, code_postal, commune, 
            len(validation_errors) == 0
        ])
        
        if validation_errors:
            for error in validation_errors:
                st.error(f"❌ {error}")
        
        if not tous_champs_remplis:
            st.info("📝 Veuillez remplir correctement tous les champs pour effectuer une prédiction.")
        
        if st.button("🚀 Prédire la valeur foncière", 
                    type="primary", 
                    use_container_width=True,
                    disabled=not tous_champs_remplis):
            
            with st.spinner("Calcul de la prédiction en cours..."):
                try:
                    # =========== NOUVELLE LOGIQUE SIMPLIFIÉE ===========
                    # Utilisation de la méthode app.py qui fonctionne
                    
                    ligne = pd.DataFrame([{
                        "Nature_mutation": "Vente",
                        "Code_postal": int(code_postal),
                        "Commune": str(commune),
                        "Code_departement": int(code_departement),
                        "Type_local": "Maison",
                        "Surface_reelle_bati": float(surface_reelle_bati),
                        "Nombre_pieces_principales": int(nombre_pieces_principales),
                        "Surface_terrain": float(surface_terrain),
                        "Moyenne_Taux": float(moyenne_taux)
                    }])
                    
                    # Conversion catégorielle comme app.py
                    categorical_cols = ["Nature_mutation", "Commune", "Type_local"]
                    for col in categorical_cols:
                        ligne[col] = ligne[col].astype("category")
                    
                    # Prédiction directe
                    prediction = model.predict(ligne)[0]
                    
                    # =========== AFFICHAGE DES RÉSULTATS ===========
                    
                    st.success(f"✅ Prédiction réalisée avec succès avec le modèle {model_choice.upper()}!")
                    st.info("Méthode utilisée: Logique app.py intégrée")
                    
                    # Main metrics
                    st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                    col_result1, col_result2 = st.columns(2)
                    
                    with col_result1:
                        st.metric(
                            label="💰 Valeur Foncière Estimée",
                            value=f"{prediction:,.0f} €",
                            help=f"Estimation basée sur le modèle {model_choice.upper()}"
                        )
                    
                    with col_result2:
                        prix_m2 = prediction / surface_reelle_bati
                        st.metric(
                            label="📐 Prix au m²",
                            value=f"{prix_m2:,.0f} €/m²",
                            help="Prix par mètre carré habitable"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Additional insights
                    st.markdown("---")
                    st.subheader("📊 Analyse détaillée")
                    
                    col_analysis1, col_analysis2 = st.columns(2)
                    
                    with col_analysis1:
                        st.markdown("**🏠 Caractéristiques du bien:**")
                        st.write(f"• Surface habitable: {surface_reelle_bati:,.0f} m²")
                        st.write(f"• Nombre de pièces: {nombre_pieces_principales}")
                        st.write(f"• Surface terrain: {surface_terrain:,.0f} m²")
                        
                        # Price per room
                        prix_par_piece = prediction / nombre_pieces_principales
                        st.write(f"• Prix par pièce: {prix_par_piece:,.0f} €")
                    
                    with col_analysis2:
                        st.markdown("**📍 Localisation:**")
                        st.write(f"• Département: {code_departement}")
                        st.write(f"• Code postal: {code_postal}")
                        st.write(f"• Commune: {commune}")
                        st.write(f"• Taux d'intérêt: {moyenne_taux}%")
                    
                    # Market context
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown("**💡 Contexte de marché:**")
                    
                    if prix_m2 > 5000:
                        st.write("🔥 Zone à prix élevé - marché premium")
                    elif prix_m2 > 3000:
                        st.write("📈 Zone à prix modéré-élevé")
                    elif prix_m2 > 2000:
                        st.write("📊 Zone à prix modéré")
                    else:
                        st.write("💚 Zone à prix accessible")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Data visualization simplifié
 #                   with st.expander("🔍 Données utilisées pour la prédiction"):
  #                      st.subheader("DataFrame d'entrée")
  #                      st.dataframe(ligne, use_container_width=True)
  #                      
  #                      st.subheader("Types de données")
  #                      dtypes_info = pd.DataFrame({
  #                          'Colonne': ligne.columns,
  #                          'Type': [str(dtype) for dtype in ligne.dtypes],
   #                         'Valeur': [ligne[col].iloc[0] for col in ligne.columns]
  #                      })
  #                      st.dataframe(dtypes_info, use_container_width=True)
                        
   #                     st.success("✅ Données traitées avec la méthode app.py")
                
                except Exception as e:
                    st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
                    
                    with st.expander("🔧 Informations de diagnostic"):
                        st.write("**Erreur:**", str(e))
                        st.write("**Type d'erreur:**", type(e).__name__)
                        
                        try:
                            import traceback
                            st.code(traceback.format_exc())
                        except:
                            pass


else:
    # Error states
    if geo_df.empty:
        st.error("❌ Données géographiques non disponibles")
        st.info("Vérifiez la présence du fichier 'Liste_Departement_Ville_ValeursFoncieres-all_fitre.csv'")
    
    if model is None:
        st.error(f"❌ Modèle {model_choice.upper()} non disponible")
        st.info(f"Vérifiez la présence du fichier 'model_{model_choice}.pkl'")