"""Convert NASA CEA thermo.inp (NASA-9 polynomials, about 2,000 species) to Cantera YAML.

thermo.inp ships with the NASA CEA program (https://www1.grc.nasa.gov/research-and-engineering/ceaweb/)
and with the `rocketcea` Python package. Usage:

    python -m thermo.cea_convert thermo.inp cea_nasa9.yaml

writes cea_nasa9.yaml and cea_nasa9.yaml.phase (name TAB gas|condensed), the two files that
`cea_reader.CEADatabase` reads. Charged species and reactant-only records (no polynomial) are skipped.
"""
import re, sys

CASE = {}   # normalise CEA's ALL-CAPS element symbols
for s in ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
          "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
          "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
          "Fr Ra Ac Th Pa U D").split():
    CASE[s.upper()] = s

def fnum(s):
    s = s.strip().replace('D', 'E').replace('d', 'E')
    return float(s) if s else 0.0

def parse(path):
    lines = open(path, errors='replace').read().splitlines()
    i, out = 0, []
    while i < len(lines):
        L = lines[i]
        # a record header starts in column 1 with a non-blank, non-comment char
        if not L[:1].strip() or L.startswith('!') or L[:1] == ' ':
            i += 1; continue
        name = L[:18].strip()
        if name.lower() in ('thermo', 'end'):
            i += 1; continue
        if i + 1 >= len(lines): break
        h = lines[i+1]
        try:
            nint = int(h[0:2])
        except ValueError:
            i += 1; continue
        phase = h[51:52].strip()
        comp, charged = {}, False
        for k in range(5):
            f = h[10+8*k:18+8*k]
            el, cnt = f[:2].strip(), f[2:].strip()
            if not el or not cnt: continue
            c = fnum(cnt)
            if c == 0: continue
            if el.upper() == 'E': charged = True; continue
            comp[CASE.get(el.upper(), el.capitalize())] = c
        if nint == 0:                      # reactant-only single point
            i += 3; continue
        rows, Tr, j = [], [], i+2
        bad = False
        for _ in range(nint):
            if j+2 >= len(lines): bad = True; break
            a = lines[j]
            try:
                tlo, thi = fnum(a[1:11]), fnum(a[11:21])
            except Exception:
                bad = True; break
            b, c = lines[j+1], lines[j+2]
            co = [fnum(b[16*k:16*k+16]) for k in range(5)]
            co += [fnum(c[0:16]), fnum(c[16:32]), fnum(c[48:64]), fnum(c[64:80])]
            rows.append(co)
            if not Tr: Tr.append(tlo)
            Tr.append(thi)
            j += 3
        i = j
        if bad or charged or not comp: continue
        out.append(dict(name=name, composition=comp, gas=(phase == '0'),
                        Tr=Tr, data=rows))
    seen={}
    for x in out:
        n=x['name']
        if n in seen:
            seen[n]+=1; x['name']=n+'_%d'%seen[n]
        else: seen[n]=1
    return out

def yaml_species(sp):
    c = ", ".join("%s: %g" % (k, v) for k, v in sp['composition'].items())
    s  = "- name: %s\n  composition: {%s}\n  thermo:\n    model: NASA9\n" % (sp['name'], c)
    s += "    reference-pressure: 1 bar\n"
    s += "    temperature-ranges: [%s]\n" % ", ".join("%.3f" % t for t in sp['Tr'])
    s += "    data:\n"
    for row in sp['data']:
        s += "    - [" + ", ".join("%.8e" % v for v in row) + "]\n"
    return s

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    sp = parse(src)
    g = sum(1 for x in sp if x['gas'])
    with open(dst, 'w') as f:
        f.write("# Converted from NASA CEA thermo.inp (McBride, Zehe & Gordon, NASA/TP-2002-211556)\n")
        f.write("phases: []\nspecies:\n")
        for x in sp: f.write(yaml_species(x))
    with open(dst + '.phase', 'w') as f:
        for x in sp: f.write("%s\t%s\n" % (x['name'], 'gas' if x['gas'] else 'condensed'))
    print("parsed %d species (%d gas, %d condensed) -> %s" % (len(sp), g, len(sp)-g, dst))
