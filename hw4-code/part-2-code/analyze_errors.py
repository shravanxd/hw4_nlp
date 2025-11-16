"""
Script to help with qualitative error analysis for Q6.
This will analyze common error types in your model's predictions.
"""

import pickle
from collections import Counter, defaultdict
from utils import load_queries_and_records

def analyze_errors(gt_sql_path, model_sql_path, gt_record_path, model_record_path):
    """
    Analyze errors in model predictions.
    """
    # Load queries and records
    gt_qs, gt_records, _ = load_queries_and_records(gt_sql_path, gt_record_path)
    model_qs, model_records, model_error_msgs = load_queries_and_records(model_sql_path, model_record_path)
    
    print("=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    
    # 1. SQL Syntax Errors
    syntax_errors = []
    for i, (msg, query) in enumerate(zip(model_error_msgs, model_qs)):
        if msg != "":
            syntax_errors.append((i, query, msg))
    
    print(f"\n1. SQL SYNTAX ERRORS: {len(syntax_errors)}/{len(model_qs)}")
    print("-" * 80)
    if syntax_errors:
        for i, (idx, query, msg) in enumerate(syntax_errors[:5]):  # Show first 5
            print(f"\nExample {i+1} (Index {idx}):")
            print(f"  Model Query: {query}")
            print(f"  Error: {msg}")
            print(f"  Ground Truth: {gt_qs[idx]}")
    
    # 2. Incorrect Results (no syntax error but wrong records)
    incorrect_results = []
    for i, (gt_rec, model_rec, msg) in enumerate(zip(gt_records, model_records, model_error_msgs)):
        if msg == "" and set(gt_rec) != set(model_rec):
            incorrect_results.append((i, model_qs[i], gt_qs[i], len(gt_rec), len(model_rec)))
    
    print(f"\n\n2. INCORRECT RESULTS (Valid SQL but wrong records): {len(incorrect_results)}/{len(model_qs)}")
    print("-" * 80)
    if incorrect_results:
        for i, (idx, model_q, gt_q, gt_len, model_len) in enumerate(incorrect_results[:5]):
            print(f"\nExample {i+1} (Index {idx}):")
            print(f"  Model Query: {model_q}")
            print(f"  Ground Truth: {gt_q}")
            print(f"  GT Records: {gt_len}, Model Records: {model_len}")
    
    # 3. Analyze error patterns
    print(f"\n\n3. ERROR PATTERN ANALYSIS")
    print("-" * 80)
    
    # Check for common SQL clause differences
    clause_errors = defaultdict(int)
    for i, (gt_q, model_q, msg) in enumerate(zip(gt_qs, model_qs, model_error_msgs)):
        if msg == "" and gt_q != model_q:
            gt_upper = gt_q.upper()
            model_upper = model_q.upper()
            
            # WHERE clause
            if 'WHERE' in gt_upper and 'WHERE' not in model_upper:
                clause_errors['Missing WHERE'] += 1
            elif 'WHERE' not in gt_upper and 'WHERE' in model_upper:
                clause_errors['Extra WHERE'] += 1
            
            # JOIN
            if 'JOIN' in gt_upper and 'JOIN' not in model_upper:
                clause_errors['Missing JOIN'] += 1
            elif 'JOIN' not in gt_upper and 'JOIN' in model_upper:
                clause_errors['Extra JOIN'] += 1
            
            # GROUP BY
            if 'GROUP BY' in gt_upper and 'GROUP BY' not in model_upper:
                clause_errors['Missing GROUP BY'] += 1
            elif 'GROUP BY' not in gt_upper and 'GROUP BY' in model_upper:
                clause_errors['Extra GROUP BY'] += 1
            
            # ORDER BY
            if 'ORDER BY' in gt_upper and 'ORDER BY' not in model_upper:
                clause_errors['Missing ORDER BY'] += 1
            elif 'ORDER BY' not in gt_upper and 'ORDER BY' in model_upper:
                clause_errors['Extra ORDER BY'] += 1
    
    print("\nClause-level errors:")
    for error_type, count in sorted(clause_errors.items(), key=lambda x: -x[1]):
        print(f"  {error_type}: {count}")
    
    # 4. Correct predictions
    correct = sum(1 for gt_q, model_q in zip(gt_qs, model_qs) if gt_q == model_q)
    print(f"\n\n4. CORRECT SQL QUERIES: {correct}/{len(model_qs)} ({100*correct/len(model_qs):.2f}%)")
    
    # 5. Partial match (correct records despite different SQL)
    partial_correct = 0
    for gt_rec, model_rec, gt_q, model_q in zip(gt_records, model_records, gt_qs, model_qs):
        if gt_q != model_q and set(gt_rec) == set(model_rec):
            partial_correct += 1
    
    print(f"   Semantically correct (different SQL, same records): {partial_correct}/{len(model_qs)}")
    
    print("\n" + "=" * 80)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_sql', type=str, default='data/dev.sql')
    parser.add_argument('--model_sql', type=str, required=True,
                       help='Path to model predictions (e.g., results/t5_ft_experiment_dev.sql)')
    parser.add_argument('--gt_records', type=str, default='records/ground_truth_dev.pkl')
    parser.add_argument('--model_records', type=str, required=True,
                       help='Path to model records (e.g., records/t5_ft_experiment_dev.pkl)')
    
    args = parser.parse_args()
    
    analyze_errors(args.gt_sql, args.model_sql, args.gt_records, args.model_records)

if __name__ == "__main__":
    main()
