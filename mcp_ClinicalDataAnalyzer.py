import pandas as pd
import numpy as np
import logging
from fastmcp import FastMCP
#per creare il server per esporre i tool MCP (da funzioni a tool)
from typing import Dict, Any, Optional
#per definire tipi di dato per chiarezza visiva
import requests
#per scaricare il dataset pubblico
import io
#per convertire il contenuto scaricato in un DataFrame

# Inizializza il server MCP
mcp = FastMCP("Clinical Data Analyzer")
#creiamo l'istanza del server mcp. 

# Configurazione del dataset pubblico (Heart Disease da UCI) https://archive.ics.uci.edu/dataset/45/heart%2Bdisease?
PUBLIC_DATASET_CONFIG = {
    "id": "heart_disease",
    "name": "Heart Disease Dataset",
    "source": "UCI Machine Learning Repository[citation:1][citation:5]",
    "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "columns": [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ],
    "description": "Dati clinici su malattie cardiache - Cleveland dataset (303 pazienti)"
}

class ClinicalDataAnalyzer:
    def __init__(self, data_path: Optional[str] = None):
        """
        Inizializza l'analyzer. Se non viene fornito un data_path,
        scarica automaticamente il dataset pubblico dall'UCI.
        """
        self.logger = self._setup_logging()

        try:
            if data_path:
                # Carica da file locale se specificato
                self.logger.info(f"Caricamento dati da file locale: {data_path}")
                self.df = pd.read_csv(data_path)
            else:
                # Scarica il dataset pubblico dall'UCI[citation:1][citation:5]
                self.logger.info("Scaricamento dataset pubblico dall'UCI")
                response = requests.get(PUBLIC_DATASET_CONFIG['url'])
                response.raise_for_status()

                self.df = pd.read_csv(
                    io.StringIO(response.text),
                    names=PUBLIC_DATASET_CONFIG['columns'],
                    na_values='?'  # Gestisce valori mancanti rappresentati da '?'
                )
                self.logger.info(f"Dataset scaricato: {self.df.shape}")

            # Analisi iniziale automatica
            self._initial_analysis()

        except Exception as e:
            self.logger.error(f"Errore durante l'inizializzazione: {e}")
            raise

    def _setup_logging(self):
        "Configura il sistema di logging"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _initial_analysis(self):
        """Esegue analisi iniziale automatica sui dati caricati"""
        self.logger.info("ANALISI INIZIALE AUTOMATICA")
        self.logger.info(f"Dimensioni dataset: {self.df.shape[0]} righe, {self.df.shape[1]} colonne")
        self.logger.info(f"Colonne: {list(self.df.columns)}")

        # Valori mancanti
        missing_percent = (self.df.isnull().sum() / len(self.df)) * 100
        self.logger.info("Percentuale valori mancanti per colonna:")
        for col, percent in missing_percent.items():
            if percent > 0:
                self.logger.info(f"  {col}: {percent:.2f}%")

        total_missing = self.df.isnull().sum().sum()
        self.logger.info(f"TOTALE VALORI MANCANTI: {total_missing}")
        self.logger.info("FINE ANALISI INIZIALE")


# Crea un'istanza globale dell'analyzer con i dati pubblici
try:
    analyzer = ClinicalDataAnalyzer()  
    logger = logging.getLogger(__name__)
    logger.info(f"Server MCP inizializzato con dataset: {PUBLIC_DATASET_CONFIG['name']}")
except Exception as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Impossibile inizializzare l'analyzer: {e}")
    analyzer = None

# Tool MCP per analisi dati preliminare
@mcp.tool()
async def analyze_clinical_data(sample_size: Optional[int] = None) -> Dict[str, Any]:
    """Analisi preliminare dei dati clinici scaricati dal dataset 
    Args:
        sample_size: Numero opzionale di record da campionare
    Returns:
        Dict con statistiche descrittive dei dati
    """
    if analyzer is None or analyzer.df is None:
        return {"status": "error", "error": "Analyzer non inizializzato correttamente"}

    try:
        df = analyzer.df.copy()
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        # Statistiche di base
        stats = {
            "dataset": PUBLIC_DATASET_CONFIG["name"],
            "source": PUBLIC_DATASET_CONFIG["source"],
            "total_records": len(df),
            "sample_size_used": sample_size if sample_size else "all",
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "missing_values_total": int(df.isnull().sum().sum()),
            "missing_values_by_column": df.isnull().sum().astype(int).to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "basic_stats": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        return {
            "status": "success",
            "analysis": stats
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Tool MCP per analisi esplorativa + avanzata
@mcp.tool()
async def exploratory_data_analysis(sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Analisi completa dei dati clinici dal dataset pubblico UCI
    
    Args:
        sample_size: Numero opzionale di record da campionare
    """
    if analyzer is None or analyzer.df is None:
        return {"status": "error", "error": "Analyzer non inizializzato correttamente"}

    try:
        df = analyzer.df.copy()
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        # Analisi dei valori unici per ogni colonna
        unique_analysis = {}
        for col in df.columns:
            unique_count = df[col].nunique()
            unique_analysis[col] = {
                "unique_count": unique_count,
                "unique_values": df[col].unique().tolist() if unique_count <= 20 else f"{unique_count} valori unici",
                #inpostiamo come valori unici solo se sono 20 o meno, altrimenti indichiamo il numero totale
                "most_frequent": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None
            }
        
        # Analisi correlazioni (solo per colonne numeriche)
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr().to_dict() if not numeric_df.empty else {}
        
        # Calcolo skewness e kurtosis
        skewness = numeric_df.skew().to_dict() if not numeric_df.empty else {}
        kurtosis = numeric_df.kurtosis().to_dict() if not numeric_df.empty else {}
        
        return {
            "status": "success",
            "dataset_info": {
                "name": PUBLIC_DATASET_CONFIG["name"],
                "source": PUBLIC_DATASET_CONFIG["source"],
                "records_analyzed": len(df),
                "sample_size": sample_size if sample_size else "all"
            },
            "exploratory_analysis": {
                "data_shape": df.shape,
                "unique_values_analysis": unique_analysis,
                "correlation_matrix": correlation_matrix,
                "skewness": skewness,
                "kurtosis": kurtosis
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Tool MCP per rilevamento problemi dati
@mcp.tool()
async def detect_data_issues(sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Rilevamento automatico di problemi nei dati
    """
    if analyzer is None or analyzer.df is None:
        return {"status": "error", "error": "Analyzer non inizializzato correttamente"}

    try:
        df = analyzer.df.copy()
        
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        issues = []
        recommendations = []
        
        # Controllo valori mancanti
        missing_percent = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_percent[missing_percent >= 5].index.tolist()
        # Consideriamo critico se >5% dei valori sono mancanti in una colonna
        if high_missing_cols:
            issues.append(f"Colonne con >5% valori mancanti: {high_missing_cols}")
            recommendations.append("Considerare la rimozione di queste colonne o eseguire imputazione avanzata")
        
        # Controllo duplicati
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Trovati {duplicate_count} record duplicati")
            recommendations.append("Valutare la rimozione dei duplicati")
        
        # Controllo outlier per colonne numeriche
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_issues = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(df)) * 100
                outlier_issues[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": float(outlier_percentage)
                }
        
        if outlier_issues:
            issues.append(f"Outlier rilevati in: {list(outlier_issues.keys())}")
            recommendations.append("Analizzare gli outlier per determinare se sono errori o valori validi")
        
        # Controllo consistenza dati clinici
        consistency_issues = []
        
        # Controlli specifici per dataset cardiaco
        if 'age' in df.columns:
            invalid_age = df[(df['age'] < 0) | (df['age'] > 120)]
            if len(invalid_age) > 0:
                consistency_issues.append(f"Valori di età non realistici: {len(invalid_age)} record")
        
        if 'trestbps' in df.columns:
            invalid_bp = df[(df['trestbps'] < 50) | (df['trestbps'] > 300)]
            if len(invalid_bp) > 0:
                consistency_issues.append(f"Valori di pressione sanguigna non realistici: {len(invalid_bp)} record")
        
        if 'chol' in df.columns:
            invalid_chol = df[(df['chol'] < 50) | (df['chol'] > 600)]
            if len(invalid_chol) > 0:
                consistency_issues.append(f"Valori di colesterolo non realistici: {len(invalid_chol)} record")
        
        if consistency_issues:
            issues.extend(consistency_issues)
            recommendations.append("Verificare la qualità dei dati e considerare la pulizia")
        
        return {
            "status": "success",
            "dataset": PUBLIC_DATASET_CONFIG["name"],
            "records_analyzed": len(df),
            "data_quality_report": {
                "total_issues_found": len(issues),
                "issues": issues,
                "recommendations": recommendations,
                "missing_values_analysis": missing_percent.round(2).to_dict(),
                "outlier_analysis": outlier_issues,
                "duplicate_records": int(duplicate_count)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# Tool MCP per ottenere informazioni sul dataset
@mcp.tool()
async def get_dataset_info() -> Dict[str, Any]:
    
    return {
        "status": "success",
        "dataset_config": PUBLIC_DATASET_CONFIG,
        "analyzer_status": "inizializzato" if analyzer and analyzer.df is not None else "non inizializzato",
        "current_data_shape": analyzer.df.shape if analyzer and analyzer.df is not None else None,
        "available_tools": [
            "analyze_clinical_data",
            "exploratory_data_analysis",
            "detect_data_issues",
            "get_dataset_info"
        ]
    }

# Tool MCP per informazioni sull'analyzer
@mcp.tool()
async def get_analyzer_info() -> Dict[str, Any]:
    """
    Restituisce informazioni sul sistema di analisi dati clinici
    """
    return {
        "system_name": "Clinical Data Analyzer MCP Server",
        "version": "1.0.0",
        "dataset_source": "UCI Machine Learning Repository - Heart Disease Dataset[citation:1][citation:5]",
        "capabilities": [
            "Analisi dati preliminare",
            "Analisi esplorativa completa",
            "Rilevamento problemi dati",
            "Statistiche descrittive",
            "Analisi correlazioni",
            "Rilevamento outlier"
        ],
        "supported_operations": [
            "analyze_clinical_data [sample_size: opzionale]",
            "exploratory_data_analysis [sample_size: opzionale]",
            "detect_data_issues [sample_size: opzionale]",
            "get_dataset_info",
            "get_analyzer_info"
        ],
        
    }

if __name__ == "__main__":
    # Avvia il server MCP
    mcp.run(transport="stdio")