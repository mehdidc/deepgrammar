from clize import run

from lightjob.cli import load_db

from deepgrammar.grammar import grammar
from deepgrammar.samplers import get_architecture_from_code

def clean():
    db = load_db()
    jobs = db.all_jobs()
    for j in jobs:
        if 'architecture' in j:
            continue
        archi = get_architecture_from_code(j['content']['codes']['classifier'])
        grammar.parse(archi)
        db.job_update(j['summary'], {'architecture': archi})


if __name__ == '__main__':
    run([clean])
