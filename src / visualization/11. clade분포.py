#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H3N2 Clade Analysis - Simple Version (No Complex Dependencies)
===============================================================

Simple clade distribution analysis without pandas/matplotlib dependencies.
Uses only built-in Python modules for maximum compatibility.

Usage:
    python clade_analysis_simple.py
"""

import csv
import re
import os
from collections import Counter, defaultdict
from datetime import datetime

# File paths

TEST_PATH = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\01. TEST_KJC.csv"
VAL_PATH  = r"C:\Users\KDT-24\final-project\Influenza_A_H3N2\#Last_Korea+China+Japan\02. VAL_KJC.csv"


def extract_year_from_seqname(seqname):
    """Extract year from seqName with robust fallback logic."""
    if not seqname:
        return None
    
    # Primary extraction: look for pattern like "/2023|"
    year_match = re.search(r'/(\d{4})\|', seqname)
    if year_match:
        year = int(year_match.group(1))
        if 2005 <= year <= 2025:
            return year
    
    # Secondary extraction: use last field after splitting by '|'
    parts = seqname.split('|')
    if parts:
        last_part = parts[-1]
        year_match = re.search(r'(\d{4})', last_part)
        if year_match:
            year = int(year_match.group(1))
            if 2005 <= year <= 2025:
                return year
    
    return None

def extract_country_from_seqname(seqname):
    """Extract country information from seqName."""
    if not seqname:
        return "Unknown"
    
    # Convert to uppercase for case-insensitive matching
    seqname_upper = seqname.upper()
    
    # Country patterns
    if any(pattern in seqname_upper for pattern in ['KOREA', 'KOBE', 'SEOUL', 'BUSAN']):
        return "Korea"
    elif any(pattern in seqname_upper for pattern in ['CHINA', 'BEIJING', 'SHANGHAI', 'GUANGDONG']):
        return "China"
    elif any(pattern in seqname_upper for pattern in ['JAPAN', 'TOKYO', 'OSAKA', 'YOKOHAMA']):
        return "Japan"
    else:
        return "Unknown"

def read_csv_file(file_path, file_label):
    """Read CSV file and extract relevant data."""
    print(f"📁 Reading {file_label} data from: {file_path}")
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Try semicolon delimiter first, then comma
            try:
                reader = csv.DictReader(file, delimiter=';')
                rows = list(reader)
            except:
                file.seek(0)
                reader = csv.DictReader(file, delimiter=',')
                rows = list(reader)
            
            if not rows:
                print(f"⚠️  No data found in {file_label}")
                return data
            
            print(f"📊 Found {len(rows)} rows in {file_label}")
            
            for i, row in enumerate(rows):
                if i % 500 == 0 and i > 0:
                    print(f"  Processed {i} rows...")
                
                # Extract seqName
                seqname = row.get('seqName', '')
                if not seqname:
                    continue
                
                # Extract clade
                clade = row.get('clade', '')
                if not clade:
                    continue
                
                # Extract year
                year = extract_year_from_seqname(seqname)
                if not year:
                    continue
                
                # Extract country
                country = extract_country_from_seqname(seqname)
                
                # QC filter if available
                qc_status = row.get('qc.overallStatus', 'good')
                if qc_status != 'good':
                    continue
                
                data.append({
                    'seqName': seqname,
                    'clade': clade,
                    'year': year,
                    'country': country
                })
        
        print(f"✅ Successfully processed {len(data)} samples from {file_label}")
        return data
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    except Exception as e:
        print(f"❌ Error reading {file_label}: {str(e)}")
        return []

def analyze_clade_basic(data, dataset_name):
    """Basic clade distribution analysis without external libraries."""
    print(f"\n🔍 Analyzing clade distribution for {dataset_name}...")
    
    if not data:
        print(f"⚠️  No data available for {dataset_name}")
        return {}
    
    # Count clades
    clade_counts = Counter(item['clade'] for item in data)
    total_samples = len(data)
    
    print(f"\n📊 {dataset_name} Basic Statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Unique clades: {len(clade_counts)}")
    
    # Sort by count
    sorted_clades = sorted(clade_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Top 10 Clades in {dataset_name}:")
    for i, (clade, count) in enumerate(sorted_clades[:10], 1):
        percentage = (count / total_samples) * 100
        print(f"  {i:2d}. {clade}: {count:4d} samples ({percentage:5.1f}%)")
    
    return dict(sorted_clades)

def analyze_temporal_patterns(data, dataset_name):
    """Analyze temporal patterns in clade distribution."""
    print(f"\n📈 Analyzing temporal patterns for {dataset_name}...")
    
    if not data:
        return
    
    # Group by year and clade
    year_clade = defaultdict(lambda: defaultdict(int))
    years = set()
    
    for item in data:
        year = item['year']
        clade = item['clade']
        year_clade[year][clade] += 1
        years.add(year)
    
    years = sorted(years)
    
    print(f"\n📅 Year Range: {min(years)} - {max(years)}")
    print(f"📊 Total Years: {len(years)}")
    
    # Find top clades
    all_clades = Counter(item['clade'] for item in data)
    top_clades = [clade for clade, _ in all_clades.most_common(8)]
    
    print(f"\n📈 Top {len(top_clades)} Clades Over Time:")
    
    # Create simple text-based visualization
    for clade in top_clades:
        print(f"\n  {clade}:")
        for year in years:
            count = year_clade[year].get(clade, 0)
            if count > 0:
                print(f"    {year}: {count:3d} samples")
    
    # Calculate clade persistence
    print(f"\n🕐 Clade Persistence Analysis:")
    for clade in top_clades:
        clade_years = []
        for year in years:
            if year_clade[year].get(clade, 0) > 0:
                clade_years.append(year)
        
        if clade_years:
            persistence = max(clade_years) - min(clade_years) + 1
            print(f"  {clade}: {persistence} years ({min(clade_years)}-{max(clade_years)})")

def analyze_geographical_patterns(data, dataset_name):
    """Analyze geographical patterns in clade distribution."""
    print(f"\n🌍 Analyzing geographical patterns for {dataset_name}...")
    
    if not data:
        return
    
    # Group by country and clade
    country_clade = defaultdict(lambda: defaultdict(int))
    countries = set()
    
    for item in data:
        country = item['country']
        clade = item['clade']
        country_clade[country][clade] += 1
        countries.add(country)
    
    print(f"\n🌏 Countries/Regions Found: {sorted(countries)}")
    
    # Create simple cross-tabulation
    print(f"\n📊 Country-Clade Distribution:")
    for country in sorted(countries):
        clade_counts = country_clade[country]
        total_country = sum(clade_counts.values())
        
        print(f"\n  {country} (Total: {total_country} samples):")
        
        # Sort clades by count
        sorted_clades = sorted(clade_counts.items(), key=lambda x: x[1], reverse=True)
        
        for clade, count in sorted_clades[:5]:  # Top 5 clades per country
            percentage = (count / total_country) * 100
            print(f"    {clade}: {count:3d} ({percentage:4.1f}%)")
        
        if len(sorted_clades) > 5:
            print(f"    ... and {len(sorted_clades)-5} more clades")
    
    return dict(country_clade)

def compare_datasets(test_data, val_data):
    """Compare TEST and VAL datasets."""
    print(f"\n🔍 Comparing TEST vs VAL datasets...")
    
    if not test_data or not val_data:
        print("⚠️  Cannot compare - missing data")
        return
    
    # Basic comparison
    test_clades = Counter(item['clade'] for item in test_data)
    val_clades = Counter(item['clade'] for item in val_data)
    
    print(f"\n📊 Dataset Comparison:")
    print(f"  TEST: {len(test_data)} samples, {len(test_clades)} unique clades")
    print(f"  VAL:  {len(val_data)} samples, {len(val_clades)} unique clades")
    
    # Find common clades
    common_clades = set(test_clades.keys()) & set(val_clades.keys())
    test_only = set(test_clades.keys()) - set(val_clades.keys())
    val_only = set(val_clades.keys()) - set(test_clades.keys())
    
    print(f"\n🔗 Clade Overlap Analysis:")
    print(f"  Common clades: {len(common_clades)}")
    print(f"  TEST-only clades: {len(test_only)}")
    print(f"  VAL-only clades: {len(val_only)}")
    
    # Compare top clades
    test_top = test_clades.most_common(10)
    val_top = val_clades.most_common(10)
    
    print(f"\n🏆 Top 5 Clades Comparison:")
    print("  TEST dataset:")
    for i, (clade, count) in enumerate(test_top[:5], 1):
        pct = (count / len(test_data)) * 100
        print(f"    {i}. {clade}: {count} ({pct:.1f}%)")
    
    print("  VAL dataset:")
    for i, (clade, count) in enumerate(val_top[:5], 1):
        pct = (count / len(val_data)) * 100
        print(f"    {i}. {clade}: {count} ({pct:.1f}%)")
    
    return {
        'common_clades': len(common_clades),
        'test_only': len(test_only),
        'val_only': len(val_only)
    }

def create_simple_report(test_clades, val_clades, combined_clades, comparison_stats):
    """Create a simple text-based report."""
    print("\n" + "="*60)
    print("📊 CLADE DISTRIBUTION ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📈 Overall Statistics:")
    print(f"  TEST dataset: {sum(test_clades.values())} total samples")
    print(f"  VAL dataset: {sum(val_clades.values())} total samples")
    print(f"  Combined: {sum(combined_clades.values())} total samples")
    print(f"  Unique clades: {len(combined_clades)}")
    
    print(f"\n🏆 Top 10 Most Common Clades (Combined):")
    top_10 = sorted(combined_clades.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (clade, count) in enumerate(top_10, 1):
        percentage = (count / sum(combined_clades.values())) * 100
        print(f"  {i:2d}. {clade}: {count:5d} samples ({percentage:5.1f}%)")
    
    print(f"\n🔍 Dataset Overlap:")
    print(f"  Common clades: {comparison_stats['common_clades']}")
    print(f"  TEST-only clades: {comparison_stats['test_only']}")
    print(f"  VAL-only clades: {comparison_stats['val_only']}")
    
    print(f"\n📊 Coverage Analysis:")
    if top_10:
        top_5_count = sum(count for _, count in top_10[:5])
        top_5_pct = (top_5_count / sum(combined_clades.values())) * 100
        print(f"  Top 5 clades cover: {top_5_pct:.1f}% of all samples")
        
        top_10_count = sum(count for _, count in top_10)
        top_10_pct = (top_10_count / sum(combined_clades.values())) * 100
        print(f"  Top 10 clades cover: {top_10_pct:.1f}% of all samples")
    
    print(f"\n📁 Generated files:")
    print("   - clade_analysis_report.txt (this report)")
    print("   - clade_distribution_summary.csv")
    
    return top_10

def save_results_to_csv(test_clades, val_clades, combined_clades, test_data, val_data):
    """Save results to CSV files for further analysis."""
    print(f"\n💾 Saving results to CSV files...")
    
    try:
        # Save clade counts
        with open('clade_distribution_summary.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Clade', 'TEST_Count', 'VAL_Count', 'Combined_Count', 
                           'TEST_Pct', 'VAL_Pct', 'Combined_Pct'])
            
            all_clades = set(combined_clades.keys())
            total_test = sum(test_clades.values())
            total_val = sum(val_clades.values())
            total_combined = sum(combined_clades.values())
            
            for clade in sorted(all_clades):
                test_count = test_clades.get(clade, 0)
                val_count = val_clades.get(clade, 0)
                combined_count = combined_clades.get(clade, 0)
                
                test_pct = (test_count / total_test * 100) if total_test > 0 else 0
                val_pct = (val_count / total_val * 100) if total_val > 0 else 0
                combined_pct = (combined_count / total_combined * 100) if total_combined > 0 else 0
                
                writer.writerow([clade, test_count, val_count, combined_count,
                               f"{test_pct:.2f}", f"{val_pct:.2f}", f"{combined_pct:.2f}"])
        
        # Save temporal data
        year_clade_data = defaultdict(lambda: defaultdict(int))
        for item in test_data + val_data:
            year_clade_data[item['year']][item['clade']] += 1
        
        with open('clade_temporal_data.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            years = sorted(year_clade_data.keys())
            all_clades = sorted(set(clade for year_data in year_clade_data.values() 
                                  for clade in year_data.keys()))
            
            writer.writerow(['Year'] + all_clades)
            for year in years:
                row = [year]
                for clade in all_clades:
                    row.append(year_clade_data[year].get(clade, 0))
                writer.writerow(row)
        
        print("✅ Results saved to CSV files!")
        
    except Exception as e:
        print(f"❌ Error saving CSV files: {str(e)}")

def main():
    """Main analysis function."""
    print("🚀 Starting H3N2 Clade Distribution Analysis (Simple Version)")
    print("="*60)
    print("Using built-in Python modules only - no external dependencies")
    print("="*60)
    
    # Read data
    test_data = read_csv_file(TEST_PATH, "TEST")
    val_data = read_csv_file(VAL_PATH, "VAL")
    
    if not test_data and not val_data:
        print("❌ No data could be loaded. Exiting.")
        return
    
    # Combine data
    combined_data = test_data + val_data
    
    print(f"\n📊 Data Summary:")
    print(f"  TEST samples: {len(test_data)}")
    print(f"  VAL samples: {len(val_data)}")
    print(f"  Combined samples: {len(combined_data)}")
    
    # Basic clade analysis
    test_clades = analyze_clade_basic(test_data, "TEST")
    val_clades = analyze_clade_basic(val_data, "VAL")
    combined_clades = analyze_clade_basic(combined_data, "COMBINED")
    
    # Temporal analysis
    analyze_temporal_patterns(combined_data, "COMBINED")
    
    # Geographical analysis
    analyze_geographical_patterns(combined_data, "COMBINED")
    
    # Compare datasets
    comparison_stats = compare_datasets(test_data, val_data)
    
    # Create report
    top_clades = create_simple_report(test_clades, val_clades, combined_clades, comparison_stats)
    
    # Save results
    save_results_to_csv(test_clades, val_clades, combined_clades, test_data, val_data)
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"   Check the generated CSV files for detailed results!")

if __name__ == "__main__":
    main()
