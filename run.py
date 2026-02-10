import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from data_loader import SkillDataLoader
from skillgenome import SkillGenome


def build_region_insights(regional, archetypes):
    insights = []
    for region, info in regional.items():
        total = info['population']
        if total == 0:
            continue
        top_idx = int(info['top_clusters'].argmax())
        top_name = archetypes[top_idx]
        top_share = info['top_clusters'][top_idx] / total
        diversity = info['diversity']
        skill_level = info['skill_level']
        if top_share >= 0.4:
            concentration = "high concentration"
        elif top_share >= 0.25:
            concentration = "moderate concentration"
        else:
            concentration = "balanced mix"
        if diversity >= 0.6:
            diversity_note = "diverse skill portfolio"
        elif diversity >= 0.4:
            diversity_note = "mixed skill distribution"
        else:
            diversity_note = "specialized skill profile"
        insights.append(
            f"{region} is identified as a skill hub with {concentration} in {top_name}. "
            f"It shows a {diversity_note} (diversity={diversity:.2f}) and an overall skill level of {skill_level:.2f}."
        )
    return insights


def build_risk_zones(regional, n_clusters, required_dist=None, shortage_threshold=0.05):
    if required_dist is None:
        required_dist = {i: 1 / n_clusters for i in range(n_clusters)}
    risk_rows = []
    for region, info in regional.items():
        total = info['population']
        if total == 0:
            continue
        region_dist = info['top_clusters'] / total
        deficits = []
        for i in range(n_clusters):
            gap = required_dist.get(i, 0) - region_dist[i]
            if gap > shortage_threshold:
                deficits.append(gap)
        risk_score = float(sum(deficits))
        if risk_score >= 0.25:
            risk_level = "High"
        elif risk_score >= 0.12:
            risk_level = "Medium"
        elif risk_score > 0:
            risk_level = "Low"
        else:
            risk_level = "Stable"
        risk_rows.append({
            "region": region,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "deficit_clusters": len(deficits)
        })
    risk_rows.sort(key=lambda x: x["risk_score"], reverse=True)
    return risk_rows


def build_global_insights(gaps, trend_df, source_choice):
    insights = []
    shortage = [k for k, v in gaps.items() if v['status'] == 'shortage']
    surplus = [k for k, v in gaps.items() if v['status'] == 'surplus']
    if shortage:
        insights.append(f"Detected {len(shortage)} skill shortages in the {source_choice} dataset.")
    if surplus:
        insights.append(f"Detected {len(surplus)} skill surpluses in the {source_choice} dataset.")

    emerging = trend_df[trend_df['trend'] == 'Emerging']
    declining = trend_df[trend_df['trend'] == 'Declining']
    if not emerging.empty:
        top = emerging.sort_values('delta_pct', ascending=False).iloc[0]
        insights.append(f"Top emerging skill: {top['skill']} (+{top['delta_pct']}%).")
    if not declining.empty:
        top = declining.sort_values('delta_pct').iloc[0]
        insights.append(f"Top declining skill: {top['skill']} ({top['delta_pct']}%).")
    return insights

def main():
    st.set_page_config(page_title="SkillGenome X", page_icon="", layout="wide")
    
    st.title("SkillGenome X - National Skill Intelligence System")
    st.markdown("**🔗 Multi-Platform Skill Inference** | GitHub + Stack Overflow")
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        import os
        github_token = os.getenv("GITHUB_API")
        use_real = True
        fast_mode = True
        use_cache = True
        
        n_people = st.slider("Number of Individuals", 100, 1000, 500, 50)
        n_clusters = st.slider("Number of Skill Clusters", 4, 12, 8)
        epochs = st.slider("Training Epochs", 20, 100, 50, 10)
        
        run_btn = st.button("Run Analysis", type="primary")
        
    
    if not run_btn:
        st.info("👈 Configure parameters in the sidebar and click 'Run Analysis' to start")
        return
    
    # 1. Load data
    with st.spinner("Loading skill signals from multiple platforms..."):
        cache_path = f"data/github_cache_{n_people}.csv" if use_real and use_cache else None
        loader = SkillDataLoader(
        n_users=n_people
        )
        data_sources = loader.load_all()

    st.success(
        f"✓ Loaded {len(data_sources['github'])} GitHub users and "
        f"{len(data_sources['stack_overflow'])} Stack Overflow users"
    )

    source_choice = st.radio(
        "Choose data source for analysis",
        ["GitHub", "Stack Overflow"],
        horizontal=True
    )
    source_key = "github" if source_choice == "GitHub" else "stack_overflow"
    data = data_sources[source_key]
    st.info(f"Analyzing {source_choice} dataset (sources are kept separate).")
    
    # Adversarial Detection Summary
    suspicious_count = data['is_suspicious'].sum()
    clean_count = len(data) - suspicious_count
    avg_trust = data['trust_score'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Clean Profiles", clean_count, delta=f"{clean_count/len(data)*100:.1f}%")
    with col2:
        st.metric("⚠️ Suspicious Profiles", suspicious_count, delta=f"{suspicious_count/len(data)*100:.1f}%", delta_color="inverse")
    with col3:
        st.metric("🛡️ Avg Trust Score", f"{avg_trust:.1f}/100")
    
    if suspicious_count > 0:
        with st.expander(f"🔍 View {suspicious_count} Suspicious Profiles"):
            suspicious_df = data[data['is_suspicious']][['username', 'trust_score', 'suspicious_flags', 'followers', 'stars', 'so_reputation']]
            st.dataframe(suspicious_df, use_container_width=True)
    
    # 2. Preprocess
    with st.spinner("Preprocessing data..."):
        # Filter suspicious profiles for clean training
        clean_data = data[~data['is_suspicious']].copy()
        if len(clean_data) < len(data):
            st.info(f"🧹 Filtered {len(data) - len(clean_data)} suspicious profiles for model training")
        
        sg = SkillGenome(latent_dim=32, n_clusters=n_clusters)
        X, features = sg.preprocess(clean_data)
    st.success(f"✓ Prepared {len(features)} features from {len(clean_data)} clean profiles")
    
    # 3. Train model
    with st.spinner(f"Training latent skill encoder for {epochs} epochs..."):
        progress_bar = st.progress(0)
        sg.train_model(X, epochs=epochs)
        progress_bar.progress(100)
    st.success("✓ Training complete")
    
    # 4. Get embeddings
    with st.spinner("Generating skill embeddings..."):
        embeddings = sg.get_embeddings(X)
    st.success(f"✓ Generated {embeddings.shape[0]} x {embeddings.shape[1]} embeddings")
    
    # 5. Analyze
    with st.spinner("Analyzing clusters and patterns..."):
        clusters, archetypes = sg.cluster_skills()
        regional = sg.regional_analysis(clean_data)
        gaps = sg.detect_gaps()
        embeddings_2d = sg.reduce_2d()
        current_dist, timeline = sg.forecast_trends(months=12)
    st.success("✓ Analysis complete")
    
    st.markdown("---")
    
    # Display results in tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Skill Archetypes",
        "Regional Analysis",
        "Skill Gaps",
        "Skill Evolution",
        "Policy Simulation",
        "Data Export"
    ])
    
    # Tab 1: Skill Archetypes
    with tab1:
        st.header("Skill Archetypes Distribution")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Cluster Statistics")
            for cid in range(sg.n_clusters):
                count = (clusters == cid).sum()
                pct = count / len(clusters) * 100
                st.metric(archetypes[cid], f"{count} people", f"{pct:.1f}%")
        
        with col2:
            # Skill Genome Map
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            colors = sns.color_palette("husl", sg.n_clusters)
            for cid in range(sg.n_clusters):
                mask = clusters == cid
                axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                               c=[colors[cid]], label=archetypes[cid], alpha=0.6, s=50, 
                               edgecolors='black', linewidth=0.5)
                centroid = embeddings_2d[mask].mean(axis=0)
                axes[0].scatter(centroid[0], centroid[1], c=[colors[cid]], s=400, marker='*',
                               edgecolors='black', linewidth=2, zorder=10)
            
            axes[0].set_title('Skill Genome Map', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Technical Depth')
            axes[0].set_ylabel('Skill Breadth')
            axes[0].legend(fontsize=8, loc='best')
            axes[0].grid(alpha=0.3)
            
            # Geographic distribution
            for cid in range(sg.n_clusters):
                mask = clusters == cid
                axes[1].scatter(clean_data.loc[mask, 'lon'], clean_data.loc[mask, 'lat'],
                               c=[colors[cid]], alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
            
            axes[1].set_title('Geographic Distribution', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Longitude')
            axes[1].set_ylabel('Latitude')
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Tab 2: Regional Analysis
    with tab2:
        st.header("Regional Skill Distribution")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Regional Profiles")
            for region, info in regional.items():
                with st.expander(f"📍 {region}"):
                    st.write(f"**Population:** {info['population']} people")
                    st.write(f"**Skill Diversity:** {info['diversity']:.2f}")
                    st.write(f"**Avg Skill Level:** {info['skill_level']:.2f}")
                    top_cluster = info['top_clusters'].argmax()
                    st.write(f"**Dominant Skill:** {archetypes[top_cluster]}")
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            regions = list(regional.keys())
            matrix = np.zeros((sg.n_clusters, len(regions)))
            for i, region in enumerate(regions):
                total = regional[region]['population']
                for cid in range(sg.n_clusters):
                    matrix[cid, i] = regional[region]['top_clusters'][cid] / total
            
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax,
                       xticklabels=regions, yticklabels=[archetypes[i] for i in range(sg.n_clusters)],
                       cbar_kws={'label': 'Proportion'})
            ax.set_title('Regional Skill Distribution Heatmap', fontsize=12, fontweight='bold')
            ax.set_xlabel('Region')
            ax.set_ylabel('Skill Cluster')
            
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("Structural Risk Zones")
        risk_rows = build_risk_zones(regional, sg.n_clusters)
        if risk_rows:
            risk_df = pd.DataFrame(risk_rows)
            st.dataframe(risk_df, use_container_width=True)
            for row in risk_rows:
                label = f"{row['region']}: {row['risk_level']} risk"
                detail = f"Risk score {row['risk_score']:.2f}, deficits {row['deficit_clusters']} clusters"
                if row['risk_level'] == "High":
                    st.error(f"{label} - {detail}")
                elif row['risk_level'] == "Medium":
                    st.warning(f"{label} - {detail}")
                elif row['risk_level'] == "Low":
                    st.info(f"{label} - {detail}")
                else:
                    st.success(f"{label} - {detail}")
        else:
            st.info("No risk zones detected for the current selection.")
    
    # Tab 3: Skill Gaps
    with tab3:
        st.header("Skill Gap Analysis")
        st.markdown("Comparing current distribution vs required distribution")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Gap Summary")
            for cid, gap_info in gaps.items():
                status = gap_info['status']
                if status == 'shortage':
                    st.error(f"🔴 **{archetypes[cid]}**: SHORTAGE ({gap_info['gap']*100:+.1f}%)")
                elif status == 'surplus':
                    st.warning(f"🟡 **{archetypes[cid]}**: SURPLUS ({gap_info['gap']*100:+.1f}%)")
                else:
                    st.success(f"🟢 **{archetypes[cid]}**: BALANCED")
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            gap_values = [float(gaps[i]['gap']) for i in range(sg.n_clusters)]
            gap_values = np.nan_to_num(gap_values, nan=0.0, posinf=0.0, neginf=0.0)
            labels = [archetypes[i] for i in range(sg.n_clusters)]
            colors_gap = ['red' if g < -0.05 else 'green' if g > 0.05 else 'gray' for g in gap_values]
            y_pos = np.arange(len(labels))
            
            ax.barh(y_pos, gap_values, color=colors_gap, alpha=0.7, edgecolor='black')
            ax.scatter(gap_values, y_pos, color='black', s=15, zorder=3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.axvline(0, color='black', linewidth=1)
            ax.axvline(-0.05, color='red', linestyle='--', alpha=0.5, label='Surplus threshold')
            ax.axvline(0.05, color='green', linestyle='--', alpha=0.5, label='Shortage threshold')
            ax.set_xlabel('Gap (Required - Current)')
            ax.set_title('Skill Gap Analysis', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.subheader("Emerging vs Declining Skills")
        delta = timeline[-1] - current_dist
        trend_rows = []
        for i in range(sg.n_clusters):
            change = float(delta[i])
            if change > 0.01:
                status = "Emerging"
            elif change < -0.01:
                status = "Declining"
            else:
                status = "Stable"
            trend_rows.append({
                "skill": archetypes[i],
                "current_pct": round(current_dist[i] * 100, 2),
                "projected_pct": round(timeline[-1][i] * 100, 2),
                "delta_pct": round(change * 100, 2),
                "trend": status
            })
        trend_df = pd.DataFrame(trend_rows)
        st.dataframe(trend_df, use_container_width=True)
    
    # Tab 4: Skill Evolution
    with tab4:
        st.header("Skill Evolution Timeline")
        st.markdown("Projected skill distribution over the next 12 months")

        months = np.arange(1, timeline.shape[0] + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(sg.n_clusters):
            ax.plot(months, timeline[:, i] * 100, label=archetypes[i])

        ax.set_xlabel("Months")
        ax.set_ylabel("Population (%)")
        ax.set_title("Skill Distribution Forecast", fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Current Distribution Baseline")
        baseline_df = pd.DataFrame({
            "Skill Cluster": [archetypes[i] for i in range(sg.n_clusters)],
            "Current Share (%)": (current_dist * 100).round(2)
        })
        st.dataframe(baseline_df, use_container_width=True)

    # Tab 5: Policy Simulation
    with tab5:
        st.header("🎯 Policy Simulation: What-If Analysis")
        st.markdown("Model interventions and see projected impact on national skill distribution")
        
        # Intervention selection
        intervention = st.radio(
            "Select Intervention Type",
            ["Upskilling Program", "New Training Initiative", "Regional Focus Campaign"],
            horizontal=False
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("⚙️ Configure Intervention")
            
            if intervention == "Upskilling Program":
                st.info("📚 Simulate retraining existing workforce from one skill to another")
                source_cluster = st.selectbox("Source Skill Cluster", 
                                             list(range(sg.n_clusters)), 
                                             format_func=lambda x: archetypes[x])
                target_cluster = st.selectbox("Target Skill Cluster", 
                                             list(range(sg.n_clusters)), 
                                             format_func=lambda x: archetypes[x])
                transfer_rate = st.slider("Transfer Rate (%)", 5, 50, 20)
                
                params = {"source": source_cluster, "target": target_cluster, "rate": transfer_rate}
                intervention_type = "upskill"
            
            elif intervention == "New Training Initiative":
                st.info("🎓 Simulate training new workforce in specific skill area")
                target_cluster = st.selectbox("Target Skill Cluster", 
                                             list(range(sg.n_clusters)), 
                                             format_func=lambda x: archetypes[x])
                new_workforce = st.slider("New Workforce Size (% of current)", 5, 30, 10)
                
                params = {"target": target_cluster, "percentage": new_workforce}
                intervention_type = "train_new"
            
            else:  # Regional Focus Campaign
                st.info("📍 Simulate regional development programs boosting specific skills")
                target_cluster = st.selectbox("Skill Cluster to Boost", 
                                             list(range(sg.n_clusters)), 
                                             format_func=lambda x: archetypes[x])
                boost_percent = st.slider("Boost Percentage", 10, 50, 25)
                
                params = {"target": target_cluster, "boost": boost_percent}
                intervention_type = "regional_focus"
            
            simulate_btn = st.button("🚀 Run Simulation", type="primary")
        
        with col2:
            if simulate_btn:
                current_dist, simulated_dist = sg.simulate_policy(intervention_type, params)
                
                st.subheader("📊 Projected Impact")
                
                # Comparison chart
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(sg.n_clusters)
                width = 0.35
                
                bars1 = ax.bar(x - width/2, current_dist * 100, width, 
                              label='Current', alpha=0.8, color='steelblue')
                bars2 = ax.bar(x + width/2, simulated_dist * 100, width, 
                              label='Projected', alpha=0.8, color='coral')
                
                ax.set_ylabel('Population (%)')
                ax.set_xlabel('Skill Clusters')
                ax.set_title('Before vs After Policy Intervention', fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels([archetypes[i] for i in range(sg.n_clusters)], 
                                   rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Impact metrics
                st.markdown("---")
                st.subheader("📈 Impact Metrics")
                
                for i in range(sg.n_clusters):
                    change = (simulated_dist[i] - current_dist[i]) * 100
                    if abs(change) > 0.5:  # Only show significant changes
                        delta_color = "normal" if change > 0 else "inverse"
                        st.metric(
                            f"{archetypes[i]}", 
                            f"{simulated_dist[i]*100:.1f}%",
                            delta=f"{change:+.1f}%",
                            delta_color=delta_color
                        )
            else:
                st.info("👈 Configure intervention parameters and click 'Run Simulation' to see projected outcomes")
    
    # Tab 6: Data Export
    with tab6:
        st.header("Export Results")
        
        results = clean_data.copy()
        results['cluster'] = clusters
        results['archetype'] = [archetypes[c] for c in clusters]
        
        st.subheader("📊 Clean Profiles Results")
        st.dataframe(results, use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = results.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="skill_results.csv",
                mime="text/csv"
            )
        
        with col2:
            st.metric("Total Records", len(results))
            st.metric("Total Features", len(results.columns))
        
        st.markdown("---")
        
        st.subheader("🛡️ Full Dataset with Adversarial Detection")
        st.dataframe(data, use_container_width=True, height=400)
        
        csv_full = data.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset (with adversarial flags)",
            data=csv_full,
            file_name="full_dataset_adversarial.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("📦 Skill Genome JSON Export")
        genome_payload = {
            "meta": {
                "n_users": int(len(clean_data)),
                "n_clusters": int(sg.n_clusters),
                "features": features,
                "timestamp": pd.Timestamp.utcnow().isoformat()
            },
            "archetypes": {str(k): v for k, v in archetypes.items()},
            "distribution": {
                "current": (current_dist * 100).round(4).tolist(),
                "projected_12m": (timeline[-1] * 100).round(4).tolist()
            },
            "gaps": gaps,
            "regional": regional,
            "emerging_declining": trend_df.to_dict(orient="records")
        }
        def _json_default(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        genome_json = json.dumps(genome_payload, indent=2, default=_json_default)
        st.download_button(
            label="📥 Download Skill Genome (JSON)",
            data=genome_json,
            file_name="skill_genome.json",
            mime="application/json"
        )

    st.markdown("---")
    st.header("Explainable Insights")
    st.markdown("Readable summaries across regions and overall trends")

    region_insights = build_region_insights(regional, archetypes)
    global_insights = build_global_insights(gaps, trend_df, source_choice)

    if region_insights:
        st.subheader("Regional Insights")
        for insight in region_insights:
            st.write(f"- {insight}")
    else:
        st.info("No regional insights available for the current selection.")

    if global_insights:
        st.subheader("Global Insights")
        for insight in global_insights:
            st.write(f"- {insight}")
    else:
        st.info("No global insights available for the current selection.")
    
    st.markdown("---")
    st.success("✅ Analysis Complete!")


if __name__ == "__main__":
    main()
