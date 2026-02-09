"""SkillGenome - Core analysis engine for inferring latent skills"""
# Uses VAE Architecture (Variational Autoencoder) to learn latent skill embeddings from GitHub user data
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

class SkillEncoder(nn.Module):
    """Neural network to learn latent skill embeddings"""
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class SkillGenome: #Orchestrator for the entire skill analysis pipeline
    """Main analysis class"""
    def __init__(self, latent_dim=32, n_clusters=8):
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        self.scaler = RobustScaler()
        self.model = None
        self.embeddings = None
        self.clusters = None
        
    def preprocess(self, df):
        """Preprocess and engineer features"""
        # Select numeric features only (exclude non-numeric like username/location)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ['id', 'lat', 'lon']]
        X = df[feature_cols].values
        
        # Handle inf/nan
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Clip extreme outliers (critical for real data!)
        for i in range(X.shape[1]):
            q99 = np.percentile(X[:, i], 99)
            X[:, i] = np.clip(X[:, i], 0, q99 * 2)  # Cap at 2x 99th percentile
        
        # Log transform for highly skewed features (followers, stars, earnings, etc.)
        X = np.log1p(X)  # log(1 + x) to handle zeros
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X)
        
        # Final check for any remaining issues
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        
        return torch.FloatTensor(X_scaled), feature_cols
    
    def train_model(self, X, epochs=50, lr=0.0001):
        """Train the skill encoder"""
        self.model = SkillEncoder(X.shape[1], self.latent_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        dataset = torch.utils.data.TensorDataset(X)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        print(f"Training skill encoder for {epochs} epochs...")
        for epoch in range(epochs):
            total_loss = 0
            for batch, in loader:
                recon, mu, logvar = self.model(batch)
                
                # VAE loss with reduced beta for training stability
                recon_loss = nn.functional.mse_loss(recon, batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
                loss = recon_loss + 0.1 * kl_loss  # Lower beta (0.1 vs 0.5)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")
        
        print("✓ Training complete")
    
    def get_embeddings(self, X):
        """Extract latent embeddings"""
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(X)
            self.embeddings = mu.numpy()
        return self.embeddings
    
    def cluster_skills(self):
        """Identify skill clusters"""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(self.embeddings)
        
        # Assign archetype names dynamically based on n_clusters
        base_archetypes = [
            "Technical Specialist", "Continuous Learner", 
            "Market Professional", "Creative Innovator",
            "Entrepreneur", "Hybrid Generalist",
            "Data Expert", "Builder",
            "Research Specialist", "Design Thinker",
            "Operations Expert", "Strategic Planner"
        ]
        
        archetypes = {}
        for i in range(self.n_clusters):
            if i < len(base_archetypes):
                archetypes[i] = base_archetypes[i]
            else:
                archetypes[i] = f"Skill Cluster {i+1}"
        
        return self.clusters, archetypes
    
    def regional_analysis(self, df):
        """Analyze skill distribution by region"""
        regional = {}
        for region in df['region'].unique():
            mask = df['region'] == region
            region_emb = self.embeddings[mask]
            region_clusters = self.clusters[mask]
            
            regional[region] = {
                'population': mask.sum(),
                'diversity': region_emb.std(axis=0).mean(),
                'skill_level': np.linalg.norm(region_emb.mean(axis=0)),
                'top_clusters': np.bincount(region_clusters, minlength=self.n_clusters)
            }
        
        return regional
    
    def reduce_2d(self):
        """Reduce to 2D for visualization"""
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        return tsne.fit_transform(self.embeddings)
    
    def detect_gaps(self, required_dist=None):
        """Identify skill gaps"""
        if required_dist is None:
            # Default: uniform distribution
            required_dist = {i: 1/self.n_clusters for i in range(self.n_clusters)}
        
        current_dist = np.bincount(self.clusters, minlength=self.n_clusters) / len(self.clusters)
        
        gaps = {}
        for i in range(self.n_clusters):
            gap = required_dist.get(i, 0) - current_dist[i]
            gaps[i] = {
                'current': current_dist[i],
                'required': required_dist.get(i, 0),
                'gap': gap,
                'status': 'shortage' if gap > 0.05 else ('surplus' if gap < -0.05 else 'balanced')
            }
        
        return gaps
    
    def simulate_policy(self, intervention_type, params):
        """Simulate policy interventions and return projected distribution"""
        current_dist = np.bincount(self.clusters, minlength=self.n_clusters) / len(self.clusters)
        simulated_dist = current_dist.copy()
        
        if intervention_type == "upskill":
            # Upskill: Move % from source cluster to target cluster
            source_cluster = params['source']
            target_cluster = params['target']
            transfer_rate = params['rate'] / 100.0
            
            transfer_amount = simulated_dist[source_cluster] * transfer_rate
            simulated_dist[source_cluster] -= transfer_amount
            simulated_dist[target_cluster] += transfer_amount
        
        elif intervention_type == "train_new":
            # Train new workforce: Add new people to target cluster
            target_cluster = params['target']
            new_people_pct = params['percentage'] / 100.0
            
            # Normalize to accommodate new people
            simulated_dist = simulated_dist * (1 / (1 + new_people_pct))
            simulated_dist[target_cluster] += new_people_pct / (1 + new_people_pct)
        
        elif intervention_type == "regional_focus":
            # Regional focus: Boost specific regions' skill clusters
            target_cluster = params['target']
            boost_pct = params['boost'] / 100.0
            
            boost_amount = simulated_dist[target_cluster] * boost_pct
            simulated_dist[target_cluster] += boost_amount
            # Normalize
            simulated_dist = simulated_dist / simulated_dist.sum()
        
        # Ensure valid distribution
        simulated_dist = np.clip(simulated_dist, 0, 1)
        simulated_dist = simulated_dist / simulated_dist.sum()
        
        return current_dist, simulated_dist

    def forecast_trends(self, months=12, volatility=0.02, seed=42):
        """Project skill distribution over time using small stochastic drift"""
        rng = np.random.default_rng(seed)
        current_dist = np.bincount(self.clusters, minlength=self.n_clusters) / len(self.clusters)
        timeline = []

        dist = current_dist.copy()
        for _ in range(months):
            drift = rng.normal(0, volatility, size=self.n_clusters)
            dist = np.clip(dist + drift, 0, None)
            dist = dist / dist.sum()
            timeline.append(dist.copy())

        return current_dist, np.array(timeline)
