import os
import time
import subprocess
import signal
import sys
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'gpu_lock_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_gpu_status():
    """Vérifie l'état du GPU et retourne True s'il est libre"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if "No running processes found" in result.stdout:
            return True
        # Vérifier si seuls les processus système sont présents
        if "Xorg" in result.stdout and "gnome-shell" in result.stdout:
            return True
        return False
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du GPU: {e}")
        return False

def lock_gpu():
    """Verrouille le GPU et lance l'entraînement"""
    if not check_gpu_status():
        logger.error("Le GPU n'est pas libre! Attendez qu'il soit disponible.")
        return False

    try:
        # Réserver le GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        logger.info("GPU verrouillé pour cette session")

        # Lancer l'entraînement
        logger.info("Démarrage de l'entraînement...")
        process = subprocess.Popen(['python', 'train.py'])

        try:
            # Attendre la fin du processus
            process.wait()
        except KeyboardInterrupt:
            logger.info("Arrêt de l'entraînement...")
            process.terminate()
            process.wait()
            logger.info("Entraînement arrêté proprement")
        except Exception as e:
            logger.error(f"Erreur pendant l'entraînement: {e}")
            process.terminate()
            process.wait()
            return False

        return True

    except Exception as e:
        logger.error(f"Erreur lors du verrouillage du GPU: {e}")
        return False

if __name__ == "__main__":
    logger.info("Démarrage du script de verrouillage GPU")
    if lock_gpu():
        logger.info("Entraînement terminé avec succès")
    else:
        logger.error("Échec de l'entraînement")
        sys.exit(1) 