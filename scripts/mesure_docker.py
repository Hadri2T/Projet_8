"""
Mesure d'une image Docker de l'API dans des conditions proches de Render (CPU et mémoire limités).

Pour chaque niveau de CPU (option --cpus de Docker, qui rationne le temps de calcul du conteneur) :
- démarrage : temps entre le lancement du conteneur et la première réponse de /health ;
- latence : N requêtes /predict envoyées une par une, temps lus dans les logs JSON de l'API (côté serveur) ;
- mémoire utilisée et nombre de fils (threads) du processus de l'API, après les requêtes ;
- non-régression : le score du client d'exemple doit rester 0.361.
Pas de base de données ni de réseau internet : on mesure le conteneur seul.

Utilisation (Docker doit être lancé) :
    docker build -t scoring-api:apres .
    python scripts/mesure_docker.py --image scoring-api:apres --cpus 0.1 0.5 1 --repetitions 3 --bilan data/mesure_docker_apres.json
    python scripts/mesure_docker.py --image scoring-api:apres --cpus 0.5 --requetes 160 --paralleles 8   # sous charge
"""

import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
CLIENT = (RACINE / "examples" / "client_exemple.json").read_bytes()
PORT = 8766  # 8765 est utilisé par docker compose
CONTENEUR = "mesure-docker"
UNITES = {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024}


def docker(*args):
    """Lance une commande docker et renvoie sa sortie."""
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True).stdout.strip()


def appeler(chemin, donnees=None):
    """Appelle l'API du conteneur et renvoie la réponse JSON."""
    requete = urllib.request.Request(
        f"http://localhost:{PORT}{chemin}", data=donnees, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(requete, timeout=120) as reponse:
        return json.load(reponse)


def mesurer(image, cpus, nb_requetes, paralleles):
    """Lance l'image avec `cpus` CPU et 512 Mo de RAM (comme Render Free), mesure, puis supprime le conteneur."""
    subprocess.run(["docker", "rm", "-f", CONTENEUR], capture_output=True)
    debut = time.perf_counter()
    docker("run", "-d", "--name", CONTENEUR, "--cpus", str(cpus), "--memory", "512m", "-p", f"{PORT}:7860", image)
    while True:  # on attend que l'API réponde
        try:
            appeler("/health")
            break
        except Exception:
            time.sleep(0.1)
    demarrage_s = time.perf_counter() - debut

    # Requêtes envoyées par `paralleles` clients simultanés (1 = une par une)
    debut = time.perf_counter()
    with ThreadPoolExecutor(max_workers=paralleles) as pool:
        reponses = list(pool.map(lambda _: appeler("/predict", CLIENT), range(nb_requetes)))
    debit = nb_requetes / (time.perf_counter() - debut)
    scores = {r["probabilite_defaut"] for r in reponses}

    # Fils du processus de l'API : le plus grand nombre parmi les processus du conteneur
    fils = int(docker("exec", CONTENEUR, "sh", "-c", "for p in /proc/[0-9]*; do ls $p/task | wc -l; done | sort -n | tail -1"))
    ram = docker("stats", "--no-stream", "--format", "{{.MemUsage}}", CONTENEUR).split(" / ")[0]
    ram_mo = float(ram[:-3]) * UNITES[ram[-3:]]

    # Temps côté serveur, lus dans les logs JSON de l'API (une ligne par appel)
    logs = subprocess.run(["docker", "logs", CONTENEUR], capture_output=True, text=True)
    appels = [json.loads(ligne) for ligne in (logs.stdout + logs.stderr).splitlines() if '"statut_http": 200' in ligne]
    subprocess.run(["docker", "rm", "-f", CONTENEUR], capture_output=True)
    appels = appels[5:]  # on écarte les 5 premiers appels (chauffe)

    def p95(valeurs):
        return statistics.quantiles(valeurs, n=20)[18]

    inference = [a["inference_ms"] for a in appels]
    latence = [a["latence_ms"] for a in appels]
    return {
        "cpus": cpus,
        "paralleles": paralleles,
        "demarrage_s": round(demarrage_s, 1),
        "debit_appels_s": round(debit, 1),
        "inference_mediane_ms": round(statistics.median(inference), 1),
        "inference_p95_ms": round(p95(inference), 1),
        "latence_mediane_ms": round(statistics.median(latence), 1),
        "latence_p95_ms": round(p95(latence), 1),
        "ram_mo": round(ram_mo),
        "fils": fils,
        "scores": sorted(scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Mesure d'une image Docker de l'API (CPU et mémoire limités)")
    parser.add_argument("--image", required=True, help="image à mesurer, par ex. scoring-api:apres")
    parser.add_argument("--cpus", type=float, nargs="+", default=[0.1, 0.5, 1], help="niveaux de CPU à tester")
    parser.add_argument("--requetes", type=int, default=40, help="nombre de requêtes /predict par niveau")
    parser.add_argument("--paralleles", type=int, default=1, help="nombre de clients simultanés (1 = une par une)")
    parser.add_argument("--repetitions", type=int, default=1, help="nombre d'essais par niveau (les mesures varient)")
    parser.add_argument("--bilan", help="fichier JSON où enregistrer les résultats")
    args = parser.parse_args()

    bilan = {
        "image": args.image,
        "taille_decompressee": docker("images", args.image, "--format", "{{.Size}}"),
        "taille_compressee_mo": round(int(docker("image", "inspect", "-f", "{{.Size}}", args.image)) / 1e6),
        "mesures": [],
    }
    print(f"{args.image} : {bilan['taille_decompressee']} décompressée, {bilan['taille_compressee_mo']} Mo compressée")
    for cpus in args.cpus:
        for essai in range(1, args.repetitions + 1):
            m = {**mesurer(args.image, cpus, args.requetes, args.paralleles), "essai": essai}
            bilan["mesures"].append(m)
            print(
                f"  {cpus:>4} CPU, {args.paralleles} en parallèle, essai {essai} | démarrage {m['demarrage_s']:5.1f} s"
                f" | débit {m['debit_appels_s']:5.1f} appels/s | inférence médiane {m['inference_mediane_ms']:6.1f} ms"
                f" (p95 {m['inference_p95_ms']:6.1f}) | latence médiane {m['latence_mediane_ms']:6.1f} ms"
                f" (p95 {m['latence_p95_ms']:6.1f}) | RAM {m['ram_mo']} Mo | {m['fils']} fils | scores {m['scores']}"
            )
    if args.bilan:
        Path(args.bilan).write_text(json.dumps(bilan, indent=2, ensure_ascii=False))
        print(f"Bilan écrit dans {args.bilan}")


if __name__ == "__main__":
    main()
