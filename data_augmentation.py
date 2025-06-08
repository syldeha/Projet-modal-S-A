import pandas as pd
import cv2
import numpy as np
import os
from pathlib import Path
import uuid

class DataAugmentator:
    def __init__(self, dataset_path="dataset", train_val_folder="train_val", csv_file="train_val.csv"):
        self.dataset_path = Path(dataset_path)
        self.train_val_path = self.dataset_path / train_val_folder
        self.csv_path = self.dataset_path / csv_file
        
        # Vérifier que les dossiers existent
        if not self.train_val_path.exists():
            raise FileNotFoundError(f"Le dossier {self.train_val_path} n'existe pas")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Le fichier {self.csv_path} n'existe pas")
    
    def load_csv(self):
        """Charger le fichier CSV"""
        return pd.read_csv(self.csv_path)
    
    def save_csv(self, df):
        """Sauvegarder le DataFrame dans le fichier CSV"""
        df.to_csv(self.csv_path, index=False)
    
    def generate_new_id(self):
        """Générer un nouvel identifiant unique"""
        return str(uuid.uuid4())[:8]  # 8 caractères pour un ID court
    
    def transformation_1_rotation(self, image):
        """Transformation 1: Rotation aléatoire (-15 à +15 degrés)"""
        angle = np.random.uniform(-15, 15)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                borderMode=cv2.BORDER_REFLECT)
        return rotated
    
    def transformation_2_brightness(self, image):
        """Transformation 2: Changement de luminosité"""
        # Convertir en float pour éviter le clipping
        bright_image = image.astype(np.float32)
        
        # Facteur de luminosité aléatoire entre 0.7 et 1.3
        brightness_factor = np.random.uniform(0.7, 1.3)
        bright_image = bright_image * brightness_factor
        
        # Clip les valeurs entre 0 et 255 et reconvertir en uint8
        bright_image = np.clip(bright_image, 0, 255).astype(np.uint8)
        return bright_image
    
    def transformation_3_horizontal_flip(self, image):
        """Transformation 3: Miroir horizontal"""
        return cv2.flip(image, 1)
    
    def apply_transformations(self, image):
        """Appliquer les 3 transformations et retourner une liste d'images"""
        transformations = [
            ("rotation", self.transformation_1_rotation),
            ("brightness", self.transformation_2_brightness),
            ("h_flip", self.transformation_3_horizontal_flip)
        ]
        
        transformed_images = []
        for name, transform_func in transformations:
            transformed_img = transform_func(image.copy())
            transformed_images.append((name, transformed_img))
        
        return transformed_images
    
    def process_single_image(self, image_id, image_info):
        """Traiter une seule image et créer ses variantes augmentées"""
        # Construire le chemin de l'image
        image_path = self.train_val_path / f"{image_id}.jpg"  # Ajustez l'extension si nécessaire
        
        # Essayer différentes extensions si .jpg n'existe pas
        if not image_path.exists():
            for ext in ['.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
                alt_path = self.train_val_path / f"{image_id}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        if not image_path.exists():
            print(f"Attention: Image {image_id} non trouvée dans {self.train_val_path}")
            return []
        
        # Charger l'image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Erreur: Impossible de charger l'image {image_path}")
            return []
        
        # Appliquer les transformations
        transformed_images = self.apply_transformations(image)
        
        new_rows = []
        for transform_name, transformed_img in transformed_images:
            # Générer un nouvel ID
            new_id = self.generate_new_id()
            
            # Sauvegarder l'image transformée
            new_image_path = self.train_val_path / f"{new_id}.jpg"
            cv2.imwrite(str(new_image_path), transformed_img)
            
            # Créer une nouvelle ligne pour le CSV avec le même contenu mais nouvel ID
            new_row = image_info.copy()
            new_row['id'] = new_id  # Remplacez 'id' par le nom de votre colonne d'identifiant
            new_row['original_id'] = image_id  # Optionnel: garder trace de l'image originale
            new_row['transformation'] = transform_name  # Optionnel: type de transformation
            
            new_rows.append(new_row)
            print(f"Créé: {new_id}.jpg (transformation: {transform_name})")
        
        return new_rows
    
    def augment_dataset(self):
        """Fonction principale pour augmenter tout le dataset"""
        # Charger le CSV
        df = self.load_csv()
        print(f"Dataset original: {len(df)} images")
        
        # Identifier la colonne d'ID (ajustez selon votre CSV)
        id_column = None
        possible_id_columns = ['id', 'image_id', 'filename', 'name']
        for col in possible_id_columns:
            if col in df.columns:
                id_column = col
                break
        
        if id_column is None:
            print("Colonnes disponibles:", df.columns.tolist())
            id_column = input("Entrez le nom de la colonne contenant les identifiants d'images: ")
        
        print(f"Utilisation de la colonne '{id_column}' comme identifiant")
        
        # Traiter chaque image
        all_new_rows = []
        for idx, row in df.iterrows():
            image_id = row[id_column]
            new_rows = self.process_single_image(image_id, row)
            all_new_rows.extend(new_rows)
            
            if (idx + 1) % 10 == 0:
                print(f"Traité {idx + 1}/{len(df)} images...")
        
        # Ajouter les nouvelles lignes au DataFrame
        if all_new_rows:
            new_df = pd.DataFrame(all_new_rows)
            augmented_df = pd.concat([df, new_df], ignore_index=True)
            
            # Sauvegarder le CSV mis à jour
            self.save_csv(augmented_df)
            print(f"\nAugmentation terminée!")
            print(f"Dataset original: {len(df)} images")
            print(f"Nouvelles images: {len(all_new_rows)} images")
            print(f"Dataset final: {len(augmented_df)} images")
        else:
            print("Aucune nouvelle image créée.")
    
    def verify_and_repair_dataset(self):
        """Vérifier la cohérence entre le CSV et les fichiers images, et réparer si nécessaire"""
        print("=== VÉRIFICATION ET RÉPARATION DU DATASET ===\n")
        
        # Charger le CSV
        df = self.load_csv()
        
        # Identifier la colonne d'ID
        id_column = self._get_id_column(df)
        
        # Statistiques
        total_entries = len(df)
        missing_files = []
        orphan_files = []
        corrupted_files = []
        repaired_files = []
        
        print(f"Total d'entrées dans le CSV: {total_entries}")
        
        # 1. Vérifier que chaque entrée CSV a son fichier image correspondant
        print("\n1. Vérification des fichiers images pour chaque entrée CSV...")
        
        for idx, row in df.iterrows():
            image_id = str(row[id_column])
            image_path = self._find_image_file(image_id)
            
            if image_path is None:
                missing_files.append((idx, image_id, row))
                print(f"❌ MANQUANT: {image_id}")
            else:
                # Vérifier si l'image peut être chargée
                image = cv2.imread(str(image_path))
                if image is None:
                    corrupted_files.append((idx, image_id, image_path))
                    print(f"⚠️  CORROMPU: {image_id} ({image_path})")
        
        # 2. Vérifier les fichiers orphelins (images sans entrée CSV)
        print(f"\n2. Vérification des fichiers orphelins...")
        
        csv_ids = set(str(id_val) for id_val in df[id_column].values)
        
        for image_file in self.train_val_path.glob("*"):
            if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                file_id = image_file.stem
                if file_id not in csv_ids:
                    orphan_files.append(image_file)
                    print(f"🔍 ORPHELIN: {image_file.name}")
        
        # 3. RÉPARATION - Recréer les images manquantes
        if missing_files:
            print(f"\n3. RÉPARATION: Recréation de {len(missing_files)} images manquantes...")
            
            for idx, missing_id, row in missing_files:
                success = self._recreate_missing_image(missing_id, row, df, id_column)
                if success:
                    repaired_files.append(missing_id)
                    print(f"✅ RÉPARÉ: {missing_id}")
                else:
                    print(f"❌ ÉCHEC: {missing_id}")
        
        # 4. RÉPARATION - Réparer les images corrompues
        if corrupted_files:
            print(f"\n4. RÉPARATION: Correction de {len(corrupted_files)} images corrompues...")
            
            for idx, corrupted_id, corrupted_path in corrupted_files:
                row = df.iloc[idx]
                success = self._recreate_missing_image(corrupted_id, row, df, id_column)
                if success:
                    repaired_files.append(corrupted_id)
                    print(f"✅ RÉPARÉ: {corrupted_id}")
                else:
                    print(f"❌ ÉCHEC: {corrupted_id}")
        
        # 5. RAPPORT FINAL
        print(f"\n=== RAPPORT FINAL ===")
        print(f"Entrées CSV totales: {total_entries}")
        print(f"Images manquantes trouvées: {len(missing_files)}")
        print(f"Images corrompues trouvées: {len(corrupted_files)}")
        print(f"Fichiers orphelins trouvés: {len(orphan_files)}")
        print(f"Images réparées avec succès: {len(repaired_files)}")
        
        if not missing_files and not corrupted_files:
            print("✅ PARFAIT: Tous les fichiers sont présents et valides!")
        else:
            print(f"⚠️  ATTENTION: {len(missing_files) + len(corrupted_files) - len(repaired_files)} problèmes non résolus")
        
        if orphan_files:
            print(f"\nFichiers orphelins détectés:")
            for orphan in orphan_files[:5]:  # Afficher seulement les 5 premiers
                print(f"  - {orphan.name}")
            if len(orphan_files) > 5:
                print(f"  ... et {len(orphan_files) - 5} autres")
        
        return {
            'total_entries': total_entries,
            'missing_files': len(missing_files),
            'corrupted_files': len(corrupted_files),
            'orphan_files': len(orphan_files),
            'repaired_files': len(repaired_files)
        }
    
    def _get_id_column(self, df):
        """Identifier la colonne d'ID dans le DataFrame"""
        possible_id_columns = ['id', 'image_id', 'filename', 'name']
        for col in possible_id_columns:
            if col in df.columns:
                return col
        
        print("Colonnes disponibles:", df.columns.tolist())
        return input("Entrez le nom de la colonne contenant les identifiants d'images: ")
    
    def _find_image_file(self, image_id):
        """Trouver le fichier image correspondant à un ID"""
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_path = self.train_val_path / f"{image_id}{ext}"
            if image_path.exists():
                return image_path
        return None
    
    def _recreate_missing_image(self, missing_id, row, df, id_column):
        """Recréer une image manquante en se basant sur l'image originale"""
        # Vérifier si c'est une image augmentée (a une colonne original_id)
        if 'original_id' in row and pd.notna(row['original_id']):
            original_id = str(row['original_id'])
            transformation_type = row.get('transformation', 'unknown')
            
            # Trouver l'image originale
            original_path = self._find_image_file(original_id)
            if original_path:
                # Charger l'image originale
                original_image = cv2.imread(str(original_path))
                if original_image is not None:
                    # Appliquer la transformation appropriée
                    if transformation_type == 'rotation':
                        new_image = self.transformation_1_rotation(original_image)
                    elif transformation_type == 'brightness':
                        new_image = self.transformation_2_brightness(original_image)
                    elif transformation_type == 'h_flip':
                        new_image = self.transformation_3_horizontal_flip(original_image)
                    else:
                        # Par défaut, utiliser rotation
                        new_image = self.transformation_1_rotation(original_image)
                    
                    # Sauvegarder la nouvelle image
                    new_path = self.train_val_path / f"{missing_id}.jpg"
                    return cv2.imwrite(str(new_path), new_image)
        
        # Si pas d'image originale trouvée, chercher une image similaire dans le dataset
        print(f"Recherche d'une image de référence pour {missing_id}...")
        
        # Prendre la première image valide trouvée comme base
        for _, other_row in df.iterrows():
            other_id = str(other_row[id_column])
            if other_id != missing_id:
                other_path = self._find_image_file(other_id)
                if other_path:
                    other_image = cv2.imread(str(other_path))
                    if other_image is not None:
                        # Créer une version modifiée de cette image
                        modified_image = self.transformation_1_rotation(other_image)
                        new_path = self.train_val_path / f"{missing_id}.jpg"
                        print(f"Création basée sur {other_id}")
                        return cv2.imwrite(str(new_path), modified_image)
        
        return False

def main():
    """Fonction principale"""
    try:
        # Initialiser l'augmentateur
        augmentator = DataAugmentator(
            dataset_path="dataset",
            train_val_folder="train_val", 
            csv_file="train_val.csv"
        )
        
        print("Que voulez-vous faire ?")
        print("1. Augmenter le dataset")
        print("2. Vérifier et réparer le dataset")
        print("3. Les deux (augmentation puis vérification)")
        
        choice = input("Votre choix (1/2/3): ").strip()
        
        if choice == "1":
            augmentator.augment_dataset()
        elif choice == "2":
            augmentator.verify_and_repair_dataset()
        elif choice == "3":
            print("=== PHASE 1: AUGMENTATION ===")
            augmentator.augment_dataset()
            print("\n=== PHASE 2: VÉRIFICATION ===")
            augmentator.verify_and_repair_dataset()
        else:
            print("Choix invalide")
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main() 