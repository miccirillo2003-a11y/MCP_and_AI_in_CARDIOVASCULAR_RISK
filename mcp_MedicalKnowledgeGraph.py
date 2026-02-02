"ESSENTIAL KNOWLEDGE GRAPH MCP"
import pandas as pd
import numpy as np
from fastmcp import FastMCP
from typing import Dict, List, Any
import json
from datetime import datetime
import scipy.stats as stats
import requests
#per scaricare il dataset pubblico
import io
#per convertire il contenuto scaricato in un DataFrame
# Inizializza MCP
mcp = FastMCP("Essential Knowledge Graph")
PUBLIC_DATASET_CONFIG = {
    "id": "heart_disease",
    "name": "Heart Disease Dataset",
    "source": "UCI Machine Learning Repository",
    "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "columns": [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ],
    "description": "Dati clinici su malattie cardiache - Cleveland dataset (303 pazienti)"
}
class EssentialKnowledgeGraph:
    
    def __init__(self):
        self.data = None
        self.knowledge = {
            "nodes": {},      
            "relations": []   
        }
        "Scarica automaticamente il dataset pubblico dall'UCI."
        

        try:
                # Scarica il dataset pubblico dall'UCI
                
                response = requests.get(PUBLIC_DATASET_CONFIG['url'])
                response.raise_for_status()

                self.data = pd.read_csv(
                    io.StringIO(response.text),
                    names=PUBLIC_DATASET_CONFIG['columns'],
                    na_values='?'  # valori mancanti rappresentati da '?'
                )
                

        except Exception as e:
            print(f"Errore durante l'inizializzazione: {e}")
            raise
        self._build_essential_knowledge()

    
    def _build_essential_knowledge(self):
        
        # Nodi
        important_features = ['age', 'sex', 'chol', 'target']
        
        for feat in important_features:
            if feat in self.data.columns:
                self.knowledge["nodes"][f"feature_{feat}"] = {
                    "type": "feature",
                    "name": feat,
                    "importance": self._get_feature_importance(feat),
                    "stats": {
                        "mean": float(self.data[feat].mean()) if self.data[feat].dtype in ['int64', 'float64'] else None,
                        "unique": int(self.data[feat].nunique())
                    }
                }
        
        # Relazioni / Archi
        if 'target' in self.data.columns:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
    
            for col in numeric_cols:
                if col != 'target':
                    #Spearman perchè il target è una variabile ordinale
                    corr, p_value = stats.spearmanr(self.data[col], self.data['target'])
                    
                    if not np.isnan(corr) and abs(corr) > 0.3 and p_value < 0.05:
                        # Interpretazione per target ordinale (0-4)
                        if abs(corr) > 0.4:
                            meaning = f"Strong predictor of disease severity (ρ={corr:.2f})"
                            relation_type = "strong_predictor"
                        else:
                            meaning = f"Moderate association with disease severity (ρ={corr:.2f})"
                            relation_type = "moderate_association"
                
                        self.knowledge["relations"].append({
                            "source": f"feature_{col}",
                            "target": "feature_target",
                            "type": relation_type,
                            "strength": abs(corr),
                            "direction": "positive" if corr > 0 else "negative",
                            "p_value": p_value,
                            "meaning": meaning
                        })
    
    def _get_feature_importance(self, feature: str) -> str:
        if feature in ['age', 'chol', 'target']:
            return "high"
        elif feature in ['trestbps', 'thalach']:
            return "medium"
        return "low"
    
    
    
    def find_connections(self, entity: str) -> List[Dict]:
        "Trova connessioni di un'entità nel Knowledge Graph"
        connections = []
        
        for rel in self.knowledge["relations"]:
            if rel["source"] == f"feature_{entity}":
                connections.append({
                    "target": rel["target"].replace("feature_", ""),
                    "relation": rel["type"],
                    "strength": rel["strength"]
                })
            elif rel["target"] == f"feature_{entity}":
                connections.append({
                    "source": rel["source"].replace("feature_", ""),
                    "relation": rel["type"],
                    "strength": rel["strength"]
                })
        
        return connections
    
    def get_risk_factors(self) -> List[Dict]:
        "Identifica i 3 principali fattori di rischio"
        if 'target' not in self.data.columns:
            return []
        
        factors = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'target':
                corr = float(self.data[[col, 'target']].corr().iloc[0, 1])
                if abs(corr) > 0.2:  # Soglia bassa per vedere più fattori
                    factors.append({
                        "feature": col,
                        "correlation": corr,
                        "risk": "increases risk" if corr > 0 else "decreases risk"
                    })
        
        # Ordina per importanza
        factors.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return factors[:3]  # Solo top 3
    
    def suggest_analysis(self) -> Dict:
        "Suggerisce UN'analisi basata sui dati"
        if not self.data.empty:
            # Controlla se ci sono dati mancanti
            missing = self.data.isnull().sum().sum()
            
            if missing > 0:
                return {
                    "priority": "high",
                    "action": "check_missing_data",
                    "reason": f"{missing} valori mancanti trovati",
                    "tool": "basic_analysis"
                }
            
            # Altrimenti suggerisce analisi correlazioni
            return {
                "priority": "medium",
                "action": "analyze_correlations",
                "reason": "Dati puliti, analizza correlazioni con target",
                "tool": "explore_relations"
            }
        
        return {"priority": "low", "action": "load_data", "reason": "Nessun dato caricato"}

# Inizializza Knowledge Graph
kg = EssentialKnowledgeGraph()


@mcp.tool()
async def explore_relations(feature: str = "target") -> Dict[str, Any]:
    """
    Esplora come una feature è connessa alle altre nel Knowledge Graph
    """
    if kg.data is None:
        return {"error": "Prima carica i dati con load_heart_disease_data"}
    
    try:
        connections = kg.find_connections(feature)
        
        # Statistiche base della feature
        stats = {}
        if feature in kg.data.columns:
            col_data = kg.data[feature]
            stats = {
                "type": str(col_data.dtype),
                "mean": float(col_data.mean()) if col_data.dtype in ['int64', 'float64'] else None,
                "unique_values": int(col_data.nunique())
            }
        
        return {
            "feature": feature,
            "stats": stats,
            "connections": connections,
            "interpretation": f"Questa feature ha {len(connections)} connessioni significative"
        }
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def clinical_insights() -> Dict[str, Any]:
    "Insights clinici essenziali dal Knowledge Graph"
    if kg.data is None:
        return {"error": "Prima carica i dati con load_heart_disease_data"}
    
    try:
        # Fattori di rischio principali
        risk_factors = kg.get_risk_factors()
        
        # Suggerimento analisi
        suggestion = kg.suggest_analysis()
        return {
            "risk_factors": risk_factors,
            "suggestion": suggestion
        }
    except Exception as e:
        return {"error": str(e)}

def _find_most_connected(self):
    "Trova la feature più connessa"
    if not kg.knowledge["relations"]:
        return "none"
    
    # Conta connessioni per ogni feature
    connections = {}
    for rel in kg.knowledge["relations"]:
        source = rel["source"].replace("feature_", "")
        target = rel["target"].replace("feature_", "")
        
        connections[source] = connections.get(source, 0) + 1
        connections[target] = connections.get(target, 0) + 1
    
    if connections:
        return max(connections, key=connections.get)
    return "none"



@mcp.tool()
async def kg_status() -> Dict[str, Any]:
    "Stato del Knowledge Graph"
    return {
        "status": "ready" if kg.data is not None else "needs_data",
        "data_loaded": kg.data is not None,
        "data_shape": kg.data.shape if kg.data is not None else None,
        "knowledge_size": {
            "nodes": len(kg.knowledge["nodes"]),
            "relations": len(kg.knowledge["relations"])
        },
        "available_tools": [
            "load_heart_disease_data - Carica dati e costruisce conoscenza",
            "explore_relations - Esplora connessioni",
            "clinical_insights - Insights clinici essenziali",
            "kg_status - Stato del sistema"
        ]
    }

if __name__ == "__main__":
    # Avvia il server MCP
    mcp.run(transport="stdio")