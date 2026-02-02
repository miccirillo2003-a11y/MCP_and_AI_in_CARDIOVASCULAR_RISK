import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging
from fastmcp import FastMCP
from typing import Dict, Any, Optional, Union
import requests
import io

# Inizializza il server MCP
mcp = FastMCP("Heart Disease Dataset ETL Processor")

# Configurazione del dataset pubblico (solo HEART DISEASE)
PUBLIC_DATASET_CONFIG = {
    "id": "heart_disease",
    "name": "Heart Disease Dataset",
    "source": "UCI Machine Learning Repository",
    "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "columns": [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ],
    "description": "Dati clinici su malattie cardiache - Cleveland dataset (303 pazienti)",
    "notes": "Valori mancanti rappresentati come '?' nell'originale"
}

class HeartDiseaseETL:
    def __init__(self):
        self.df = None
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        "Configuriamo il sistema di logging"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def fetch_dataset(self) -> pd.DataFrame:
        """
        Scarica il dataset Heart Disease dall'UCI Repository
        """
        self.logger.info(f"Scaricando dataset: {PUBLIC_DATASET_CONFIG['name']}")
        
        try:
            # Effettua la richiesta per ottenere i dati
            response = requests.get(PUBLIC_DATASET_CONFIG['url'])
            response.raise_for_status()  # Solleva eccezione in caso di errore
            
            # Legge i dati CSV
            self.df = pd.read_csv(
                io.StringIO(response.text),
                names=PUBLIC_DATASET_CONFIG['columns'],
                na_values='?'
            )
            #StringIO trasforma una stringa in un oggetto file-like.
            #Tratta i dati grezzi come se fossero un file (csv).
            self.logger.info(f"Dataset scaricato con successo: {self.df.shape}")
            self.logger.info(f"Colonne: {list(self.df.columns)}")
            
            return self.df
            
        except requests.RequestException as e:
            self.logger.error(f"Errore nel download del dataset: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Errore nel parsing del dataset: {e}")
            raise
    # Parsing dei dati è il processo di analizzare e convertire dati in un formato strutturato e utilizzabile.
    def transform(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Pulizia, trasformazione e feature engineering
        """
        self.logger.info("Iniziando fase TRANSFORM")
        
        if self.df is None:
            self.logger.error("Nessun dato disponibile. Esegui fetch_dataset() prima.")
            return None
        
        # Campiona i dati se richiesto e se serve
        if sample_size and sample_size < len(self.df):
            self.df = self.df.sample(n=sample_size, random_state=42)
            self.logger.info(f"Campionati {sample_size} record")
        
        # Salva le colonne originali
        self.original_columns = set(self.df.columns)
        
        # Analisi iniziale
        self._initial_analysis()
        
        # Gestione valori mancanti
        self._handle_missing_values()
        
        # Feature engineering
        self._feature_engineering()
        
        # Codifica variabili categoriche
        self._encode_categorical_variables()
        
        # Normalizzazione features numeriche
        self._normalize_numerical_features()
        
        # Verifica finale
        self._final_quality_check()
        
        self.logger.info("Fase TRANSFORM completata")
        return self.df
    
    def _initial_analysis(self):
        
        self.logger.info("ANALISI INIZIALE DATASET ")
        self.logger.info(f"Dimensioni: {self.df.shape[0]} righe, {self.df.shape[1]} colonne")
        
        # Statistiche base
        missing_percent = (self.df.isnull().sum() / len(self.df)) * 100
        self.logger.info(f"Valori mancanti per colonna:\n{missing_percent.round(2)}")
        
        # Distribuzione target
        if 'target' in self.df.columns:
            target_dist = self.df['target'].value_counts()
            self.logger.info(f"Distribuzione target (0=no disease, 1-4=disease):\n{target_dist}")
        
        # Tipi di dato
        self.logger.info(f"Tipi di dato:\n{self.df.dtypes}")
    
    def _handle_missing_values(self):
        
        self.logger.info("Gestione valori mancanti:")
        
        total_missing_before = self.df.isnull().sum().sum()
        self.logger.info(f"Valori mancanti totali prima: {total_missing_before}")
        
        # Strategie di imputazione specifiche per heart disease
        imputation_strategies = {
            'numeric_mean': ['trestbps', 'chol', 'thalach', 'oldpeak'],
            'numeric_median': ['ca'],  # ca ha outliers
            'categorical_mode': ['thal']
        }
        
        for strategy, columns in imputation_strategies.items():
            for col in columns:
                if col in self.df.columns and self.df[col].isnull().sum() > 0:
                    if strategy == 'numeric_mean':
                        value = self.df[col].mean()
                        self.df[col].fillna(value, inplace=True)
                    elif strategy == 'numeric_median':
                        value = self.df[col].median()
                        self.df[col].fillna(value, inplace=True)
                    elif strategy == 'categorical_mode':
                        value = self.df[col].mode()[0]
                        self.df[col].fillna(value, inplace=True)
                    self.logger.info(f"Imputato {col} con {strategy.split('_')[1]}")
        
        total_missing_after = self.df.isnull().sum().sum()
        self.logger.info(f"Valori mancanti totali dopo: {total_missing_after}")
    
    def _feature_engineering(self):
        """Creazione di nuove feature cliniche"""
        self.logger.info("Feature Engineering:")
        
        new_features = []
        
        # Categorizzazione età
        if 'age' in self.df.columns:
            self.df['age_group'] = pd.cut(
                self.df['age'],
                bins=[20, 40, 50, 60, 80],
                labels=['young', 'middle_aged', 'senior', 'elderly']
            )
            new_features.append('age_group')
        
        # Categorizzazione pressione sanguigna
        if 'trestbps' in self.df.columns:
            self.df['bp_category'] = pd.cut(
                self.df['trestbps'],
                bins=[0, 120, 130, 140, 180, 250],
                labels=['optimal', 'normal', 'high_normal', 
                       'hypertension1', 'hypertension2']
            )
            new_features.append('bp_category')
        
        # Categorizzazione colesterolo
        if 'chol' in self.df.columns:
            self.df['chol_category'] = pd.cut(
                self.df['chol'],
                bins=[0, 200, 240, 300, 600],
                labels=['desirable', 'borderline', 'high', 'very_high']
            )
            new_features.append('chol_category')
        
        # Punteggio di rischio semplificato
        risk_factors = []
        if 'age' in self.df.columns:
            risk_factors.append((self.df['age'] > 55).astype(int))
        if 'trestbps' in self.df.columns:
            risk_factors.append((self.df['trestbps'] > 140).astype(int))
        if 'chol' in self.df.columns:
            risk_factors.append((self.df['chol'] > 240).astype(int))
        
        if risk_factors:
            self.df['risk_score'] = sum(risk_factors)
            new_features.append('risk_score')
        
        # Target binario (malattia vs no malattia)
        if 'target' in self.df.columns:
            self.df['has_disease'] = (self.df['target'] > 0).astype(int)
            new_features.append('has_disease')
        
        if new_features:
            self.logger.info(f"Feature create: {new_features}")
    
    def _encode_categorical_variables(self):
        """Codifica delle variabili categoriche"""
        self.logger.info("Codifica variabili categoriche...")
        
        # Colonne categoriche nel dataset heart disease
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category').cat.codes
                self.logger.info(f"Codificata: {col}")
        
        # Codifica le nuove feature categoriche create
        new_categorical = ['age_group', 'bp_category', 'chol_category']
        for col in new_categorical:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category').cat.codes
    
    def _normalize_numerical_features(self):
        
        self.logger.info("Normalizzazione feature numeriche:")
        
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        for col in numerical_cols:
            if col in self.df.columns:
                # Normalizzazione min-max (0-1)
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                if max_val > min_val:  # Evita divisione per zero
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
                    self.logger.info(f"Normalizzata: {col}")
    
    def _final_quality_check(self):
        """Verifica finale della qualità dei dati"""
        self.logger.info("VERIFICA FINALE QUALITÀ DATI:")
        
        # Verifica valori mancanti
        missing_total = self.df.isnull().sum().sum()
        if missing_total == 0:
            self.logger.info("✓ Nessun valore mancante")
        else:
            self.logger.warning(f"⚠️ Ancora {missing_total} valori mancanti")
            self.df = self.df.dropna()
        
        # Verifica duplicati
        duplicates = self.df.duplicated().sum()
        if duplicates == 0:
            self.logger.info("✓ Nessun duplicato")
        else:
            self.logger.info(f"Rimossi {duplicates} duplicati")
            self.df = self.df.drop_duplicates()
        
        # Statistiche finali
        self.logger.info(f"Dimensioni finali: {self.df.shape}")
        self.logger.info(f"Colonne finali: {list(self.df.columns)}")
        
        # Informazioni sulle nuove feature
        final_columns = set(self.df.columns)
        added_features = final_columns - self.original_columns
        if added_features:
            self.logger.info(f"Feature aggiunte: {list(added_features)}")


@mcp.tool()
async def fetch_and_process_heart_data(sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Scarica e processa il dataset Heart Disease dall'UCI
    Args:
        sample_size: Numero di record da campionare (opzionale, se None usa tutto)
    Returns:
        Dati processati e statistiche
    """
    try:
        etl = HeartDiseaseETL()
        
        # Scarica i dati
        df = etl.fetch_dataset()
        
        # Processa i dati
        processed_df = etl.transform(sample_size=sample_size)
        
        # Prepara risultato
        result = {
            "status": "success",
            "dataset": PUBLIC_DATASET_CONFIG["name"],
            "source": PUBLIC_DATASET_CONFIG["source"],
            "original_records": len(df),
            "processed_records": len(processed_df),
            "columns": list(processed_df.columns),
            "sample_size_used": sample_size if sample_size else "all",
            "data_preview": processed_df.head(10).to_dict(orient='records'),
            "statistics": {
                "age_mean": float(processed_df['age'].mean()) if 'age' in processed_df.columns else None,
                "chol_mean": float(processed_df['chol'].mean()) if 'chol' in processed_df.columns else None,
                "disease_percentage": float(processed_df['has_disease'].mean() * 100) if 'has_disease' in processed_df.columns else None
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "dataset_config": PUBLIC_DATASET_CONFIG
        }


@mcp.tool()
async def get_dataset_info() -> Dict[str, Any]:
    "Restituisce informazioni sul dataset disponibile"
    return {
        "status": "success",
        "dataset": PUBLIC_DATASET_CONFIG,
        "usage_instructions": [
            "Usa fetch_and_process_heart_data() per scaricare e processare i dati",
            "Parametro opzionale sample_size per campionare i dati",
            "I dati vengono automaticamente puliti e arricchiti con feature engineering"
        ],
        "features_original": PUBLIC_DATASET_CONFIG["columns"],
        "features_added": [
            "age_group", "bp_category", "chol_category", 
            "risk_score", "has_disease"
        ]
    }


@mcp.tool()
async def analyze_heart_disease(sample_size: Optional[int] = None) -> Dict[str, Any]:
    "Analisi del dataset e calcolo statistiche e correlazioni in base al target"
    try:
        etl = HeartDiseaseETL()
        df = etl.fetch_dataset()
        processed_df = etl.transform(sample_size=sample_size)
        
        # Analisi statistiche avanzate
        analysis = {
            "status": "success",
            "basic_stats": {
                "total_patients": len(processed_df),
                "male_percentage": float((processed_df['sex'] == 1).mean() * 100) if 'sex' in processed_df.columns else None,
                "female_percentage": float((processed_df['sex'] == 0).mean() * 100) if 'sex' in processed_df.columns else None,
                "average_age": float(processed_df['age'].mean()) if 'age' in processed_df.columns else None,
                "disease_prevalence": float((processed_df['target'] > 0).mean() * 100) if 'target' in processed_df.columns else None
            },
            "clinical_insights": {
                "high_bp_patients": float((processed_df['trestbps'] > 140).mean() * 100) if 'trestbps' in processed_df.columns else None,
                "high_chol_patients": float((processed_df['chol'] > 240).mean() * 100) if 'chol' in processed_df.columns else None,
                "patients_with_exang": float((processed_df['exang'] == 1).mean() * 100) if 'exang' in processed_df.columns else None
            },
            "risk_distribution": {
                "low_risk": float((processed_df['risk_score'] == 0).mean() * 100) if 'risk_score' in processed_df.columns else None,
                "medium_risk": float((processed_df['risk_score'] == 1).mean() * 100) if 'risk_score' in processed_df.columns else None,
                "high_risk": float((processed_df['risk_score'] >= 2).mean() * 100) if 'risk_score' in processed_df.columns else None
            }
        }
        
        # Calcola correlazioni se ci sono abbastanza dati
        if len(processed_df) > 10:
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                correlation_matrix = processed_df[numeric_cols].corr()
                # Prendi solo le correlazioni con il target
                if 'target' in numeric_cols:
                    target_corr = correlation_matrix['target'].sort_values(ascending=False)
                    analysis["correlations_with_target"] = target_corr.to_dict()
        
        return analysis
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool() 
async def export_processed_data(format: str = "json", sample_size: Optional[int] = None) -> Dict[str, Any]:
    try:
        etl = HeartDiseaseETL()
        df = etl.fetch_dataset()
        processed_df = etl.transform(sample_size=sample_size)
        
        if format.lower() == "json":
            return {
                "status": "success",
                "format": "json",
                "data": processed_df.to_dict(orient='records'),
                "metadata": {
                    "columns": list(processed_df.columns),
                    "records": len(processed_df),
                    
                }
            }
            
        elif format.lower() == "csv":
            csv_string = processed_df.to_csv(index=False)
            return {
                "status": "success", 
                "format": "csv",
                "data": csv_string,
                "records": len(processed_df),
                "columns": list(processed_df.columns)
            }
            
            
        else:
            return {
                "status": "error",
                "error": f"Formato non supportato: {format}. Usa 'json' o 'csv'."
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    # Avvia il server MCP
    mcp.run(transport="stdio")