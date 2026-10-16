import hashlib
import io
from pathlib import Path, PurePosixPath
import sys
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

SOURCES = {
    'python': (
        'Mohan-Zhang-u/PenSimPy',
        'e08fc71fef193448f3ee9a304187dc92b88f3acc',
        '0af2b57c7737c3a554369c5ebcb2eea3b7be2bb1e975dfc38a08a4bd3fb32ddb',
        'PenSimPy',
    ),
    'matlab': (
        'Lemniscabio/IndPenSim',
        '45c3d8c53ba423671acfb63de5855ecd4c8aa9d4',
        '7e66a46c7bdd5698083f2f711fd2d6d798cc7e666ee3952e607bc66cd9561545',
        'IndPenSimMATLAB',
    ),
}
MATLAB_FILES = {
    'indpensim_ode.m', 'Parameter_list.m', 'fctrl_indpensim.m',
    'PIDSimple3.m', 'createBatch.m', 'createChannel.m',
}


def python_fixes(folder):
    # same SciPy fixes used by the demo
    p = folder / 'pensimpy/peni_env_setup.py'
    s = p.read_text(encoding='utf-8')
    s = s.replace('import fastodeint',
                  'try:\n    import fastodeint\n'
                  'except ImportError:\n    fastodeint = None')
    s = s.replace('np.arange(t_start, t_end + h_ode, h_ode)',
                  'np.linspace(t_start, t_end, 41)')
    old = 'y_sol = odeint(indpensim_ode_py, x00, t_span, tfirst=True, args=(par,))'
    new = '''atol = np.full(len(x00), 1e-8)
            atol[6] = 1e-12
            y_sol, solver_info = odeint(
                indpensim_ode_py, x00, t_span, tfirst=True, args=(par,),
                mxstep=10000, full_output=True, rtol=1e-5, atol=atol)
            if solver_info['message'] != 'Integration successful.':
                raise RuntimeError(f"Step {k}: {solver_info['message']}")'''
    assert old in s
    p.write_text(s.replace(old, new), encoding='utf-8')
    p = folder / 'pensimpy/ode/indpensim_ode_py.py'
    s = p.read_text(encoding='utf-8')
    s = s.replace('    mu_p = par[0]', '    y = y.copy()\n    mu_p = par[0]')
    start = s.index('    # if inhib_flag == 2:')
    end = s.index('\n\n', start)
    lines = s[start:end].splitlines()
    block = '    if inhib_flag == 2:\n'
    block += '\n'.join('    ' + line for line in lines[1:])
    p.write_text(s[:start] + block + s[end:], encoding='utf-8')


def download(kind, root):
    repo, commit, checksum, name = SOURCES[kind]
    target = root / name
    if target.exists():
        print(f'{name} already exists, keeping it.')
        return
    print(f'Downloading {name}...')
    url = f'https://codeload.github.com/{repo}/zip/{commit}'
    with urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != checksum:
        raise RuntimeError(f'{name}: download checksum mismatch')
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temp:
        stage = Path(temp) / name
        stage.mkdir()
        with ZipFile(io.BytesIO(data)) as archive:
            for entry in archive.infolist():
                path = PurePosixPath(entry.filename)
                relative = PurePosixPath(*path.parts[1:])
                if entry.is_dir() or '..' in relative.parts or path.is_absolute():
                    continue
                if kind == 'matlab':
                    keep = relative.name in MATLAB_FILES
                else:
                    keep = (relative.parts[:1] == ('pensimpy',)
                            or str(relative) in ('LICENSE', 'README.md'))
                if keep:
                    output = stage / relative
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(archive.read(entry))
        if kind == 'python':
            python_fixes(stage)
        else:
            assert all((stage / file).is_file() for file in MATLAB_FILES)
        stage.rename(target)
    print(f'Ready: {target}')


if __name__ == '__main__':
    choice = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if choice not in ('all', 'python', 'matlab'):
        sys.exit('Usage: python download_simulators.py [python|matlab]')
    root = Path(__file__).resolve().parent / 'vendor'
    for kind in SOURCES if choice == 'all' else (choice,):
        download(kind, root)
