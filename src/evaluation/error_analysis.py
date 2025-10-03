"""
Error Analysis Module
File: src/evaluation/error_analysis.py
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorAnalyzer:
    """Analyze model errors and identify patterns"""
    
    def __init__(self, output_dir="results/evaluation"):
        self.output_dir = output_dir
        self.error_data = {}
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_errors(self, model_name, y_true, y_pred, texts=None):
        """Perform comprehensive error analysis"""
        logger.info(f"Analyzing errors for {model_name}")
        
        # Find misclassified examples
        errors = []
        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            if true_label != pred_label:
                error_entry = {
                    'index': i,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'error_type': f'{true_label}_to_{pred_label}'
                }
                
                if texts is not None and i < len(texts):
                    error_entry['text'] = texts[i]
                
                errors.append(error_entry)
        
        error_df = pd.DataFrame(errors)
        
        # Calculate error statistics
        total_samples = len(y_true)
        total_errors = len(errors)
        error_rate = total_errors / total_samples if total_samples > 0 else 0
        
        # Error patterns
        error_patterns = Counter([e['error_type'] for e in errors])
        
        # Per-class error rates
        class_errors = {}
        for label in ['negative', 'neutral', 'positive']:
            label_mask = y_true == label
            label_total = np.sum(label_mask)
            label_errors = np.sum((y_true[label_mask] != y_pred[label_mask]))
            class_errors[label] = {
                'total': int(label_total),
                'errors': int(label_errors),
                'error_rate': float(label_errors / label_total) if label_total > 0 else 0
            }
        
        analysis = {
            'total_samples': total_samples,
            'total_errors': total_errors,
            'error_rate': error_rate,
            'error_patterns': dict(error_patterns),
            'class_errors': class_errors,
            'error_dataframe': error_df
        }
        
        self.error_data[model_name] = analysis
        
        logger.info(f"{model_name} - Total errors: {total_errors}/{total_samples} ({error_rate*100:.2f}%)")
        
        return analysis
    
    def get_most_common_errors(self, model_name, top_n=5):
        """Get most common error types"""
        if model_name not in self.error_data:
            raise ValueError(f"No error data for model: {model_name}")
        
        error_patterns = self.error_data[model_name]['error_patterns']
        most_common = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        print(f"\nMost Common Errors for {model_name}:")
        for error_type, count in most_common:
            print(f"  {error_type}: {count} occurrences")
        
        return most_common
    
    def get_sample_errors(self, model_name, error_type=None, n_samples=10):
        """Get sample error instances"""
        if model_name not in self.error_data:
            raise ValueError(f"No error data for model: {model_name}")
        
        error_df = self.error_data[model_name]['error_dataframe']
        
        if error_type:
            sample = error_df[error_df['error_type'] == error_type].head(n_samples)
        else:
            sample = error_df.head(n_samples)
        
        return sample.to_dict('records')
    
    def plot_error_distribution(self, model_name):
        """Plot error distribution"""
        if model_name not in self.error_data:
            raise ValueError(f"No error data for model: {model_name}")
        
        error_patterns = self.error_data[model_name]['error_patterns']
        
        if not error_patterns:
            logger.warning(f"No errors to plot for {model_name}")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        error_types = list(error_patterns.keys())
        counts = list(error_patterns.values())
        
        bars = ax.barh(error_types, counts, color='coral')
        ax.set_xlabel('Number of Errors')
        ax.set_ylabel('Error Type')
        ax.set_title(f'Error Distribution - {model_name}')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{model_name}_error_distribution.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"Error distribution plot saved for {model_name}")
        plt.close()
    
    def plot_class_error_rates(self):
        """Plot error rates by class for all models"""
        if not self.error_data:
            logger.warning("No error data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = ['negative', 'neutral', 'positive']
        x = np.arange(len(classes))
        width = 0.8 / len(self.error_data)
        
        for i, (model_name, analysis) in enumerate(self.error_data.items()):
            error_rates = [analysis['class_errors'][cls]['error_rate'] * 100 
                          for cls in classes]
            offset = width * (i - len(self.error_data)/2 + 0.5)
            ax.bar(x + offset, error_rates, width, label=model_name)
        
        ax.set_xlabel('Sentiment Class')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title('Error Rates by Class')
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in classes])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/class_error_rates.png', 
                   dpi=300, bbox_inches='tight')
        logger.info("Class error rates plot saved")
        plt.close()
    
    def compare_model_errors(self):
        """Compare errors across models"""
        if len(self.error_data) < 2:
            logger.warning("Need at least 2 models for comparison")
            return
        
        comparison_data = []
        
        for model_name, analysis in self.error_data.items():
            comparison_data.append({
                'Model': model_name,
                'Total Errors': analysis['total_errors'],
                'Error Rate (%)': analysis['error_rate'] * 100,
                'Negative Error Rate (%)': analysis['class_errors']['negative']['error_rate'] * 100,
                'Neutral Error Rate (%)': analysis['class_errors']['neutral']['error_rate'] * 100,
                'Positive Error Rate (%)': analysis['class_errors']['positive']['error_rate'] * 100
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Error Rate (%)')
        
        print("\nError Rate Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def generate_error_report(self, model_name, n_samples=20):
        """Generate detailed error report"""
        if model_name not in self.error_data:
            raise ValueError(f"No error data for model: {model_name}")
        
        analysis = self.error_data[model_name]
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append(f"ERROR ANALYSIS REPORT - {model_name}")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-" * 70)
        report_lines.append(f"Total Samples: {analysis['total_samples']}")
        report_lines.append(f"Total Errors: {analysis['total_errors']}")
        report_lines.append(f"Error Rate: {analysis['error_rate']*100:.2f}%")
        report_lines.append("")
        
        # Per-class error rates
        report_lines.append("PER-CLASS ERROR RATES")
        report_lines.append("-" * 70)
        for cls in ['negative', 'neutral', 'positive']:
            cls_data = analysis['class_errors'][cls]
            report_lines.append(f"{cls.capitalize()}:")
            report_lines.append(f"  Total: {cls_data['total']}")
            report_lines.append(f"  Errors: {cls_data['errors']}")
            report_lines.append(f"  Error Rate: {cls_data['error_rate']*100:.2f}%")
        report_lines.append("")
        
        # Error patterns
        report_lines.append("ERROR PATTERNS")
        report_lines.append("-" * 70)
        error_patterns = sorted(analysis['error_patterns'].items(), 
                              key=lambda x: x[1], reverse=True)
        for error_type, count in error_patterns:
            percentage = (count / analysis['total_errors'] * 100) if analysis['total_errors'] > 0 else 0
            report_lines.append(f"{error_type}: {count} ({percentage:.1f}%)")
        report_lines.append("")
        
        # Sample errors
        if 'text' in analysis['error_dataframe'].columns:
            report_lines.append(f"SAMPLE ERRORS (showing up to {n_samples})")
            report_lines.append("-" * 70)
            
            sample_errors = self.get_sample_errors(model_name, n_samples=n_samples)
            for i, error in enumerate(sample_errors, 1):
                report_lines.append(f"{i}. True: {error['true_label']} | "
                                  f"Predicted: {error['predicted_label']}")
                report_lines.append(f"   Text: {error['text'][:100]}...")
                report_lines.append("")
        
        report_lines.append("=" * 70)
        
        # Save report
        report_file = f'{self.output_dir}/{model_name}_error_report.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Error report saved to {report_file}")
        
        return '\n'.join(report_lines)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    y_true = np.random.choice(['negative', 'neutral', 'positive'], size=200)
    
    # BERT predictions
    bert_pred = y_true.copy()
    error_indices = np.random.choice(len(bert_pred), size=35, replace=False)
    for idx in error_indices:
        bert_pred[idx] = np.random.choice(['negative', 'neutral', 'positive'])
    
    # Sample texts
    texts = [f"Sample text {i}" for i in range(200)]
    
    # Analyze errors
    analyzer = ErrorAnalyzer()
    analyzer.analyze_errors('BERT', y_true, bert_pred, texts)
    analyzer.plot_error_distribution('BERT')
    analyzer.get_most_common_errors('BERT')
    analyzer.generate_error_report('BERT')