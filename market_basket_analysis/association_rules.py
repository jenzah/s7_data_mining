# association_rules.py

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict

# ==============================================================================
# Load and explore data
# ==============================================================================
def load_and_prepare_data(filepath: str):
    """
    Load transaction data from CSV and prepare it for analysis.
    
    Args:
        filepath: Path to the CSV file containing transactions
        
    Returns:
        tuple: (encoded_dataframe, transactions_list)
            - encoded_dataframe (df): Binary matrix suitable for apriori/fpgrowth
            - transactions_list (list of lists): List of transactions (list of items)
    """
    # Load the file
    df = pd.read_csv(filepath)
    
    # Convert items_space_separated to list of lists
    transactions = []
    for items_string in df['items_space_separated']:
        items = items_string.strip().split()
        transactions.append(items)
    
    # Encode transactions as binary matrix
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    return df_encoded, transactions


def display_data_info(df_encoded: pd.DataFrame, transactions: List[List[str]]):
    """
    Display information about the loaded dataset.
    
    Args:
        df_encoded: Encoded transaction dataframe
        transactions: List of transactions
    """
    print("=" * 60)
    print("INFORMATIONS SUR LES DONNÉES")
    print("=" * 60)
    print(f"Nombre total de transactions : {len(transactions)}")
    print(f"Nombre total de produits uniques : {df_encoded.shape[1]}")
    print(f"\nPremières transactions :")
    for i, trans in enumerate(transactions[:5], 1):
        print(f"  Transaction {i}: {', '.join(trans)}")
    print("\n" + "=" * 60 + "\n")


def get_item_frequencies(df_encoded: pd.DataFrame):
    """
    Calculate frequency and support for each individual item.
    
    Args:
        df_encoded: Encoded transaction dataframe
        
    Returns:
        DataFrame with item frequencies sorted by support
    """
    item_counts = df_encoded.sum()
    total_transactions = len(df_encoded)
    
    freq_df = pd.DataFrame({
        'Produit': item_counts.index,
        'Fréquence': item_counts.values,
        'Support': item_counts.values / total_transactions
    })
    
    return freq_df.sort_values('Support', ascending=False).reset_index(drop=True)


# ==============================================================================
# Run algorithms
# ==============================================================================
def run_apriori(df_encoded: pd.DataFrame, min_support: float = 0.2, min_confidence: float = 0.6, display=True):
    """
    Execute Apriori algorithm to find frequent itemsets and generate association rules.
    
    Args:
        df_encoded: Encoded transaction dataframe
        min_support: Minimum support threshold (default: 0.2)
        min_confidence: Minimum confidence threshold (default: 0.6)
        
    Returns:
        tuple (df, df, float): (frequent_itemsets, rules, execution_time)
    """
    if display:
        print("=" * 60)
        print("EXÉCUTION DE L'ALGORITHME APRIORI")
        print("=" * 60)
        print(f"Support minimal : {min_support*100}%")
        print(f"Confiance minimale : {min_confidence*100}%")
        print("\nTraitement en cours...")
    
    # Measure execution time
    start_time = time.time()
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", 
                                  min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))
        # Calculate lift
        rules['lift'] = rules['lift']
    else:
        rules = pd.DataFrame()
    
    execution_time = time.time() - start_time
    
    if display:
        print(f"✓ Terminé en {execution_time:.4f} secondes")
        print(f"✓ {len(frequent_itemsets)} ensembles fréquents trouvés")
        print(f"✓ {len(rules)} règles d'association générées")
        print("=" * 60 + "\n")
    
    return frequent_itemsets, rules, execution_time


def run_fpgrowth(df_encoded: pd.DataFrame, min_support: float = 0.2, min_confidence: float = 0.6):
    """
    Execute FP-Growth algorithm to find frequent itemsets and generate association rules.
    
    Args:
        df_encoded: Encoded transaction dataframe
        min_support: Minimum support threshold (default: 0.2)
        min_confidence: Minimum confidence threshold (default: 0.6)
        
    Returns:
        tuple (df, df, float): (frequent_itemsets, rules, execution_time)
    """
    print("=" * 60)
    print("EXÉCUTION DE L'ALGORITHME FP-GROWTH")
    print("=" * 60)
    print(f"Support minimal : {min_support*100}%")
    print(f"Confiance minimale : {min_confidence*100}%")
    print("\nTraitement en cours...")
    
    # Measure execution time
    start_time = time.time()
    
    # Find frequent itemsets
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="confidence", 
                                  min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))
        # Calculate lift
        rules['lift'] = rules['lift']
    else:
        rules = pd.DataFrame()
    
    execution_time = time.time() - start_time
    
    print(f"✓ Terminé en {execution_time:.4f} secondes")
    print(f"✓ {len(frequent_itemsets)} ensembles fréquents trouvés")
    print(f"✓ {len(rules)} règles d'association générées")
    print("=" * 60 + "\n")
    
    return frequent_itemsets, rules, execution_time


# ==============================================================================
# Display results
# ==============================================================================
# Local function
def format_itemset(itemset):
    """
    Format a frozenset itemset as a readable string.
    
    Args:
        itemset: Frozenset of items
        
    Returns:
        Formatted string representation
    """
    return "{" + ", ".join(sorted(list(itemset))) + "}"


def display_frequent_itemsets(frequent_itemsets: pd.DataFrame, top_n: int = 10):
    """
    Display the most frequent itemsets - overall and with multiple items (if any).
    
    Args:
        frequent_itemsets: DataFrame of frequent itemsets
        top_n: Number of top itemsets to display
    """
    print("\n" + "=" * 60)
    print(f"TOP {top_n} ENSEMBLES FRÉQUENTS (par support)")
    print("=" * 60)
    
    # Sort by support
    sorted_itemsets = frequent_itemsets.sort_values('support', ascending=False).head(top_n)
    
    for rank, (idx, row) in enumerate(sorted_itemsets.iterrows(), 1):
        itemset_str = format_itemset(row['itemsets'])
        size = len(row['itemsets'])
        # print(f"{itemset_str:30s} | Support: {row['support']:.3f} | Taille: {size}")
        print(f"{rank:2d}. {itemset_str:26s} | Support: {row['support']:.3f} | Taille: {size}")
    
    print("=" * 60 + "\n")

def display_frequent_list_itemsets(frequent_itemsets: pd.DataFrame, top_n: int = 10):
    """
    Display the most frequent itemsets - overall and with multiple items (if any).
    
    Args:
        frequent_itemsets: DataFrame of frequent itemsets
        top_n: Number of top itemsets to display
    """
    # Now show only combinations (size > 1)
    multi_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 1)]
    
    # Get the total count, and only multiset count
    nb_total = len(frequent_itemsets)
    nb_multi = len(multi_itemsets)

    if len(multi_itemsets) > 0:
        print("=" * 60)
        print(f"TOP {top_n} COMBINAISONS DE PRODUITS (taille > 1)")
        print("=" * 60)
        
        print(f"Nombre total d'itemsets: {nb_total}")
        print(f"Nombre d'itemsets de produits combinés: {nb_multi}\n")

        sorted_multi = multi_itemsets.sort_values('support', ascending=False).head(top_n)
        print("=" * 60)
        
        for rank, (idx, row) in enumerate(sorted_multi.iterrows(), 1):
            itemset_str = format_itemset(row['itemsets'])
            size = len(row['itemsets'])
            # print(f"{itemset_str:30s} | Support: {row['support']:.3f} | Taille: {size}")
            print(f"{rank:2d}. {itemset_str:26s} | Support: {row['support']:.3f} | Taille: {size}")
        
        print("=" * 60 + "\n")
    else:
        print("\nAucune combinaison de produits trouvée (tous les itemsets sont de taille 1).\n")
        print("=" * 60 + "\n")


def display_rules(rules: pd.DataFrame, top_n: int = None, sort_by: str = 'lift'):
    """
    Display association rules as a formatted DataFrame.
    
    Args:
        rules: DataFrame of association rules
        top_n: Number of rules to display (None for all)
        sort_by: Metric to sort by ('lift', 'confidence', 'support')
        
    Returns:
        Formatted DataFrame with rules
    """
    if len(rules) == 0:
        print("Aucune règle trouvée.\n")
        return pd.DataFrame()
    
    # Create formatted dataframe
    display_df = pd.DataFrame({
        'Règle': rules.apply(lambda x: f"{format_itemset(x['antecedents'])} --> {format_itemset(x['consequents'])}", axis=1),
        'Support': rules['support'],
        'Confiance': rules['confidence'],
        'Lift': rules['lift']
    })

    column_mapping = {
        'lift': 'Lift',
        'confidence': 'Confiance',
        'support': 'Support'
    }
    
    # Sort by specified metric
    sort_column = column_mapping.get(sort_by, 'Lift')
    display_df = display_df.sort_values(sort_column, ascending=False)
    
    # Limit to top_n if specified
    if top_n:
        display_df = display_df.head(top_n)
    
    # Reset index to show ranking
    display_df = display_df.reset_index(drop=True)
    display_df.index = display_df.index + 1  # Start ranking at 1
    
    return display_df


def compare_algorithms(apriori_results: tuple, fpgrowth_results: tuple):
    """
    Compare results from Apriori and FP-Growth algorithms.
    
    Args:
        apriori_results: Tuple of (itemsets, rules, time) from Apriori
        fpgrowth_results: Tuple of (itemsets, rules, time) from FP-Growth
        
    Returns:
        DataFrame with comparison metrics
    """
    apriori_itemsets, apriori_rules, apriori_time = apriori_results
    fpgrowth_itemsets, fpgrowth_rules, fpgrowth_time = fpgrowth_results
    
    comparison = pd.DataFrame({
        'Métrique': [
            'Ensembles fréquents',
            'Règles générées',
            'Temps d\'exécution (s)',
            'Vitesse relative'
        ],
        'Apriori': [
            len(apriori_itemsets),
            len(apriori_rules),
            f"{apriori_time:.4f}",
            "1.00x"
        ],
        'FP-Growth': [
            len(fpgrowth_itemsets),
            len(fpgrowth_rules),
            f"{fpgrowth_time:.4f}",
            f"{apriori_time/fpgrowth_time:.2f}x" if fpgrowth_time > 0 else "N/A"
        ]
    })
    
    return comparison


def display_comparison(comparison_df: pd.DataFrame):
    """
    Display algorithm comparison table.
    
    Args:
        comparison_df: DataFrame with comparison metrics
    """
    print("=" * 60)
    print("COMPARAISON DES ALGORITHMES")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60 + "\n")


def analyse_rule_quality(rules: pd.DataFrame):
    """
    Analyse the quality of generated rules based on lift values.
    
    Args:
        rules: DataFrame of association rules
        
    Returns:
        Dictionary with quality statistics
    """
    if len(rules) == 0:
        return {
            'total': 0,
            'positive_correlation': 0,
            'negative_correlation': 0,
            'no_correlation': 0
        }
    
    positive = len(rules[rules['lift'] > 1])
    negative = len(rules[rules['lift'] < 1])
    neutral = len(rules[rules['lift'] == 1])
    
    return {
        'total': len(rules),
        'positive_correlation': positive,
        'negative_correlation': negative,
        'no_correlation': neutral,
        'avg_lift': rules['lift'].mean(),
        'avg_confidence': rules['confidence'].mean(),
        'avg_support': rules['support'].mean()
    }


def display_quality_analysis(quality_stats: Dict):
    """
    Display rule quality analysis.
    
    Args:
        quality_stats: Dictionary with quality statistics
    """
    print("=" * 60)
    print("ANALYSE DE LA QUALITÉ DES RÈGLES")
    print("=" * 60)
    print(f"Nombre total de règles : {quality_stats['total']}")
    print(f"\nCorrélation positive (lift > 1) : {quality_stats['positive_correlation']} règles")
    print(f"Corrélation négative (lift < 1) : {quality_stats['negative_correlation']} règles")
    print(f"Pas de corrélation (lift = 1)   : {quality_stats['no_correlation']} règles")
    
    if quality_stats['total'] > 0:
        print(f"\nMoyennes :")
        print(f"  Lift moyen       : {quality_stats['avg_lift']:.3f}")
        print(f"  Confiance moyenne: {quality_stats['avg_confidence']:.3f}")
        print(f"  Support moyen    : {quality_stats['avg_support']:.3f}")
    
    print("=" * 60 + "\n")


def generate_recommendations(rules: pd.DataFrame, min_lift: float = 1.2, top_n: int = 5):
    """
    Generate marketing recommendations based on association rules.
    
    Args:
        rules: DataFrame of association rules
        min_lift: Minimum lift value for recommendations
        top_n: Number of recommendations to generate
        
    Returns:
        List of recommendation dictionaries
    """
    if len(rules) == 0:
        return []
    
    # Filter rules with significant positive correlation
    strong_rules = rules[rules['lift'] >= min_lift].sort_values('lift', ascending=False)
    
    recommendations = []
    
    for idx, row in strong_rules.head(top_n).iterrows():
        antecedent = format_itemset(row['antecedents'])
        consequent = format_itemset(row['consequents'])
        
        rec = {
            'antecedent': antecedent,
            'consequent': consequent,
            'support': row['support'],
            'confidence': row['confidence'],
            'lift': row['lift'],
            'recommendation': f"Proposer une promotion groupée : {antecedent} + {consequent}"
        }
        recommendations.append(rec)
    
    return recommendations


def display_recommendations(recommendations: List[Dict]):
    """
    Display marketing recommendations.
    
    Args:
        recommendations: List of recommendation dictionaries
    """
    print("=" * 80)
    print("RECOMMANDATIONS MARKETING")
    print("=" * 80)
    
    if len(recommendations) == 0:
        print("Aucune recommandation générée avec les critères spécifiés.\n")
        return
    
    for i, rec in enumerate(recommendations, 1):
        print(f"📌 Recommandation {i}:")
        print(f"   Règle : {rec['antecedent']} --> {rec['consequent']}")
        print(f"   Lift : {rec['lift']:.3f} | Confiance : {rec['confidence']:.3f} | Support : {rec['support']:.3f}")
        print(f"   💡 {rec['recommendation']}\n")
    
    print("\n" + "=" * 80 + "\n")
