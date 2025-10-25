"""
Segment-based synthetic telco customer data generator.

Uses K-Means clustering to identify customer segments and learns
statistical distributions within each segment to generate realistic
synthetic data that preserves feature correlations.

Usage:
    python generate_synthetic_data.py --n_samples 1000 --output synthetic_data.csv
    python generate_synthetic_data.py --n_samples 5000 --n_segments 6 --validate
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class SegmentedTelcoGenerator:
    """
    Generates synthetic telco customer data using segment-based approach.
    This generator preserves feature correlations by learning distributions
    within customer segments rather than globally.
    """
    
    def __init__(self, reference_data_path, n_segments=4):
        """
        Initialize generator by segmenting reference data.
        Args:
            reference_data_path (str): Path to original telco dataset
            n_segments (int): Number of customer segments to identify
        """
        self.ref_data = pd.read_excel(reference_data_path, sheet_name='Telco_Churn')
        self.n_segments = n_segments
        self._create_segments()
        self._learn_segment_distributions()
        print(f"Identified and learned {n_segments} customer segments\n")
    
    def _create_segments(self):
        """Identify customer segments using K-Means clustering."""
        
        # Select features for clustering
        cluster_features = self.ref_data[[
            'Tenure Months',
            'Monthly Charges',
            'Churn Value'
        ]].copy()
        
        # Add encoded categorical features
        cluster_features['Contract_MTM'] = (self.ref_data['Contract'] == 'Month-to-month').astype(int)
        cluster_features['Internet_Fiber'] = (self.ref_data['Internet Service'] == 'Fiber optic').astype(int)
        cluster_features['Has_Security'] = (self.ref_data['Online Security'] == 'Yes').astype(int)
        
        # Handle missing values
        cluster_features = cluster_features.fillna(cluster_features.median())
        
        # Standardize features for clustering
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(cluster_features)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=self.n_segments, random_state=42, n_init=10)
        self.ref_data['Segment'] = kmeans.fit_predict(features_scaled)
        
        # Analyze segments
        print("\nSegment Analysis:")
        for seg in range(self.n_segments):
            seg_data = self.ref_data[self.ref_data['Segment'] == seg]
            churn_rate = seg_data['Churn Value'].mean()
            avg_tenure = seg_data['Tenure Months'].mean()
            mtm_pct = (seg_data['Contract'] == 'Month-to-month').mean()
            
            print(f"\nSegment {seg} (n={len(seg_data)}):")
            print(f"  Churn Rate: {churn_rate:.1%}")
            print(f"  Avg Tenure: {avg_tenure:.1f} months")
            print(f"  Month-to-month: {mtm_pct:.1%}")
    
    def _learn_segment_distributions(self):
        """Learn feature distributions within each segment."""
        
        self.segment_stats = {}
        
        for seg in range(self.n_segments):
            seg_data = self.ref_data[self.ref_data['Segment'] == seg]
            
            self.segment_stats[seg] = {
                'size': len(seg_data),
                'churn_rate': seg_data['Churn Value'].mean(),
                
                # Numeric feature statistics
                'tenure': {
                    'mean': seg_data['Tenure Months'].mean(),
                    'std': seg_data['Tenure Months'].std(),
                    'min': 0,
                    'max': 72
                },
                'monthly_charges': {
                    'mean': seg_data['Monthly Charges'].mean(),
                    'std': seg_data['Monthly Charges'].std(),
                    'min': 18,
                    'max': 120
                },
                
                # Categorical feature probabilities
                'gender': self._get_probs(seg_data, 'Gender'),
                'senior': self._get_probs(seg_data, 'Senior Citizen'),
                'partner': self._get_probs(seg_data, 'Partner'),
                'dependents': self._get_probs(seg_data, 'Dependents'),
                'phone': self._get_probs(seg_data, 'Phone Service'),
                'multiple_lines': self._get_probs(seg_data, 'Multiple Lines'),
                'internet': self._get_probs(seg_data, 'Internet Service'),
                'online_security': self._get_probs(seg_data, 'Online Security'),
                'online_backup': self._get_probs(seg_data, 'Online Backup'),
                'device_protection': self._get_probs(seg_data, 'Device Protection'),
                'tech_support': self._get_probs(seg_data, 'Tech Support'),
                'streaming_tv': self._get_probs(seg_data, 'Streaming TV'),
                'streaming_movies': self._get_probs(seg_data, 'Streaming Movies'),
                'contract': self._get_probs(seg_data, 'Contract'),
                'paperless': self._get_probs(seg_data, 'Paperless Billing'),
                'payment': self._get_probs(seg_data, 'Payment Method'),
            }
    
    def _get_probs(self, data, column):
        """Get probability distribution for a column."""
        counts = data[column].value_counts(normalize=True)
        return counts.to_dict()
    
    def _sample_categorical(self, probs_dict):
        """Sample from categorical distribution."""
        if not probs_dict:
            return None
        categories = list(probs_dict.keys())
        probabilities = list(probs_dict.values())
        return np.random.choice(categories, p=probabilities)
    
    def _generate_customer_from_segment(self, segment_id, customer_id):
        """
        Generate a single customer from a specific segment.
        Args:
            segment_id (int): Segment to generate customer from
            customer_id (str): Unique customer identifier
        Returns:
            dict: Customer record with all features
        """
        stats = self.segment_stats[segment_id]
        
        # Generate numeric features based on segment distributions
        tenure = np.clip(
            np.random.normal(stats['tenure']['mean'], stats['tenure']['std']),
            stats['tenure']['min'],
            stats['tenure']['max']
        )
        tenure = int(round(tenure))
        
        monthly_charges = np.clip(
            np.random.normal(stats['monthly_charges']['mean'], stats['monthly_charges']['std']),
            stats['monthly_charges']['min'],
            stats['monthly_charges']['max']
        )
        monthly_charges = round(monthly_charges, 2)
        
        # Calculate total charges proportional to tenure
        total_charges = monthly_charges * tenure * np.random.uniform(0.9, 1.1)
        total_charges = round(total_charges, 2)
        
        # Generate categorical features from segment distributions
        gender = self._sample_categorical(stats['gender'])
        senior_citizen = self._sample_categorical(stats['senior'])
        partner = self._sample_categorical(stats['partner'])
        dependents = self._sample_categorical(stats['dependents'])
        phone_service = self._sample_categorical(stats['phone'])
        internet_service = self._sample_categorical(stats['internet'])
        
        # Handle dependent features
        if phone_service == 'Yes':
            multiple_lines = self._sample_categorical(stats['multiple_lines'])
        else:
            multiple_lines = 'No phone service'
        
        # Internet-dependent services
        if internet_service == 'No':
            online_security = 'No internet service'
            online_backup = 'No internet service'
            device_protection = 'No internet service'
            tech_support = 'No internet service'
            streaming_tv = 'No internet service'
            streaming_movies = 'No internet service'
        else:
            online_security = self._sample_categorical(stats['online_security'])
            online_backup = self._sample_categorical(stats['online_backup'])
            device_protection = self._sample_categorical(stats['device_protection'])
            tech_support = self._sample_categorical(stats['tech_support'])
            streaming_tv = self._sample_categorical(stats['streaming_tv'])
            streaming_movies = self._sample_categorical(stats['streaming_movies'])
        
        contract = self._sample_categorical(stats['contract'])
        paperless_billing = self._sample_categorical(stats['paperless'])
        payment_method = self._sample_categorical(stats['payment'])
        
        # Generate churn based on segment's churn rate
        churn_value = np.random.binomial(1, stats['churn_rate'])
        
        return {
            'CustomerID': customer_id,
            'Gender': gender,
            'Senior Citizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'Tenure Months': tenure,
            'Phone Service': phone_service,
            'Multiple Lines': multiple_lines,
            'Internet Service': internet_service,
            'Online Security': online_security,
            'Online Backup': online_backup,
            'Device Protection': device_protection,
            'Tech Support': tech_support,
            'Streaming TV': streaming_tv,
            'Streaming Movies': streaming_movies,
            'Contract': contract,
            'Paperless Billing': paperless_billing,
            'Payment Method': payment_method,
            'Monthly Charges': monthly_charges,
            'Total Charges': total_charges,
            'Churn Value': churn_value
        }
    
    def generate(self, n_samples=1000):
        """
        Generate synthetic customer data using segment-based approach.
        Args:
            n_samples (int): Number of synthetic customers to generate
        Returns:
            pd.DataFrame: Synthetic customer data
        """        
        # Determine samples per segment (proportional to original distribution)
        segment_sizes = [self.segment_stats[i]['size'] for i in range(self.n_segments)]
        segment_proportions = np.array(segment_sizes) / sum(segment_sizes)
        samples_per_segment = (segment_proportions * n_samples).astype(int)
        
        # Adjust for rounding differences
        diff = n_samples - samples_per_segment.sum()
        samples_per_segment[0] += diff
                
        # Generate synthetic customers
        synthetic_data = []
        customer_counter = 1
        
        for seg in range(self.n_segments):
            for _ in range(samples_per_segment[seg]):
                customer_id = f"SYNTH-{customer_counter:06d}"
                customer = self._generate_customer_from_segment(seg, customer_id)
                synthetic_data.append(customer)
                customer_counter += 1
        
        df = pd.DataFrame(synthetic_data)
        print(f"\nGenerated {len(df)} synthetic customers")
        
        # Show segment characteristics
        print("\nSynthetic Data Segment Characteristics:")
        for seg in range(self.n_segments):
            seg_synthetic = df[
                (df['Tenure Months'] >= self.segment_stats[seg]['tenure']['mean'] - 10) &
                (df['Tenure Months'] <= self.segment_stats[seg]['tenure']['mean'] + 10)
            ]
            if len(seg_synthetic) > 0:
                churn_rate = seg_synthetic['Churn Value'].mean()
                print(f"Segment {seg}: Churn Rate = {churn_rate:.1%}")
        
        return df
    
    def validate(self, synthetic_df):
        """
        Validate synthetic data against reference data.
        Args:
            synthetic_df (pd.DataFrame): Generated synthetic data
        """
        print("VALIDATION: Synthetic vs. Original Data")
        
        # Overall churn rate comparison
        ref_churn = self.ref_data['Churn Value'].mean()
        syn_churn = synthetic_df['Churn Value'].mean()
        
        print(f"\nOverall Churn Rate:")
        print(f"  Reference:  {ref_churn:.3f} ({ref_churn*100:.1f}%)")
        print(f"  Synthetic:  {syn_churn:.3f} ({syn_churn*100:.1f}%)")
        print(f"  Difference: {abs(ref_churn - syn_churn):.4f}")
        
        # Tenure comparison
        print(f"\nTenure Months:")
        print(f"  Reference  - Mean: {self.ref_data['Tenure Months'].mean():.2f}, "
              f"Std: {self.ref_data['Tenure Months'].std():.2f}")
        print(f"  Synthetic  - Mean: {synthetic_df['Tenure Months'].mean():.2f}, "
              f"Std: {synthetic_df['Tenure Months'].std():.2f}")
        
        # Contract distribution comparison
        print(f"\nContract Distribution:")
        ref_contract = self.ref_data['Contract'].value_counts(normalize=True)
        syn_contract = synthetic_df['Contract'].value_counts(normalize=True)
        
        comparison = pd.DataFrame({
            'Reference': ref_contract,
            'Synthetic': syn_contract
        })
        print(comparison.round(3))
        


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Generate segment-based synthetic telco customer data'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1000,
        help='Number of synthetic customers to generate (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/synthetic/synthetic_customers_segmented.csv',
        help='Output file path'
    )
    parser.add_argument(
        '--reference',
        type=str,
        default='data/raw/Telco_customer_churn.xlsx',
        help='Reference data path'
    )
    parser.add_argument(
        '--n_segments',
        type=int,
        default=4,
        help='Number of customer segments (default: 4)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    generator = SegmentedTelcoGenerator(args.reference, n_segments=args.n_segments)
    synthetic_df = generator.generate(n_samples=args.n_samples)
    
    # Save to file
    synthetic_df.to_csv(args.output, index=False)
    print(f"\nSaved synthetic data to: {args.output}")
    
    # Run validation if requested
    if args.validate:
        generator.validate(synthetic_df)


if __name__ == '__main__':
    main()