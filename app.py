"""Streamlit app: HOMO and LUMO energies from a SMILES string, predicted by the GNN ensemble trained on CSD_EES
and, for comparison, by LightGBM on fingerprints.

    pip install -r requirements.txt
    streamlit run app.py

The trained models and reference data are in model/ (exported from the training pipeline).
"""
import pandas as pd
import streamlit as st

from predictor import Predictor, draw, draw_attributions

st.set_page_config(page_title='HOMO/LUMO predictor', page_icon='⚛', layout='wide')

EXAMPLES = {
    'Pentacene': 'c1ccc2cc3cc4cc5ccccc5cc4cc3cc2c1',
    'Perylene': 'c1cc2cccc3c4cccc5cccc(c(c1)c23)c54',
    'TCNQ': 'N#CC(C#N)=C1C=CC(=C(C#N)C#N)C=C1',
    'Triphenylamine': 'c1ccc(cc1)N(c1ccccc1)c1ccccc1',
    'Terthiophene': 'c1csc(c1)-c1ccc(s1)-c1cccs1',
    'Rubrene': 'c1ccc(cc1)-c1c2ccccc2c(-c2ccccc2)c2c(-c3ccccc3)c3ccccc3c(-c3ccccc3)c12',
}


@st.cache_resource(show_spinner='Loading the models...')
def load_predictor():
    return Predictor()


predictor = load_predictor()
scores, lgbm = predictor.info['scores'], predictor.info['lgbm']

with st.sidebar:
    st.header('About')
    st.markdown(
        f"""
Predicts the **HOMO and LUMO energies** (eV) of an organic molecule from its SMILES, as computed by DFT in the
**CSD_EES** dataset (Omar, Nematiaram, Troisi & Padula, doi:10.17638/datacat.liverpool.ac.uk/1472).

**Models**, trained on {predictor.info['n_train']:,} molecules from crystal structures:
- **GNN ensemble** (main prediction): the mean of {len(predictor.models)} graph neural networks
  ({', '.join(predictor.models)}).
- **LightGBM** (classical machine learning, for comparison): gradient-boosted trees on Avalon + ECFP4 count
  fingerprints, tuned with Optuna.

| | GNN HOMO | GNN LUMO | LightGBM HOMO | LightGBM LUMO |
|---|---|---|---|---|
| Test R² | {scores['HOMO']['test_R2']:.3f} | {scores['LUMO']['test_R2']:.3f} | {lgbm['HOMO']['R2']:.3f} | {lgbm['LUMO']['R2']:.3f} |
| Test MAE (eV) | {scores['HOMO']['test_MAE']:.3f} | {scores['LUMO']['test_MAE']:.3f} | {lgbm['HOMO']['MAE (eV)']:.3f} | {lgbm['LUMO']['MAE (eV)']:.3f} |
| 5-fold CV R² | {scores['HOMO']['cv_R2_mean']:.3f} | {scores['LUMO']['cv_R2_mean']:.3f} | {lgbm['HOMO']['cv_R2_mean']:.3f} | {lgbm['LUMO']['cv_R2_mean']:.3f} |

**Limits:** only the elements {', '.join(predictor.info['elements'])} (plus H) were in the training data;
predictions are less reliable for molecules unlike any training molecule (see the similarity shown with each
prediction). The values are single-molecule DFT energies, not measured or solid-state levels.
"""
    )

single, batch = st.tabs(['Single molecule', 'Many molecules'])

with single:
    def use_example():
        st.session_state['smiles'] = EXAMPLES[st.session_state['example']]

    if 'smiles' not in st.session_state:
        st.session_state['smiles'] = EXAMPLES['Pentacene']
    st.selectbox('Example', list(EXAMPLES), key='example', on_change=use_example)
    smiles = st.text_input('SMILES', key='smiles')
    show_atoms = st.checkbox('Show atom contributions (Integrated Gradients of the GINE model)')
    if smiles.strip():
        r = predictor.predict([smiles])[0]
        if r['error']:
            st.error(r['error'])
        else:
            left, right = st.columns([1, 1.3])
            left.image(draw(r['mol']), caption=r['smiles'])
            with right:
                st.markdown('**GNN ensemble** (main prediction)')
                c1, c2, c3 = st.columns(3)
                c1.metric('HOMO', f"{r['HOMO']:.2f} eV", help=f"spread of the {len(predictor.models)} models: ± {r['HOMO_spread']:.2f} eV")
                c2.metric('LUMO', f"{r['LUMO']:.2f} eV", help=f"spread of the {len(predictor.models)} models: ± {r['LUMO_spread']:.2f} eV")
                c3.metric('Gap (LUMO − HOMO)', f"{r['gap']:.2f} eV")
                st.markdown('**LightGBM** (classical machine learning on fingerprints)')
                c1, c2, c3 = st.columns(3)
                c1.metric('HOMO', f"{r['LGBM_HOMO']:.2f} eV", delta=f"{r['LGBM_HOMO'] - r['HOMO']:+.2f} vs GNN",
                          delta_color='off')
                c2.metric('LUMO', f"{r['LGBM_LUMO']:.2f} eV", delta=f"{r['LGBM_LUMO'] - r['LUMO']:+.2f} vs GNN",
                          delta_color='off')
                c3.metric('Gap (LUMO − HOMO)', f"{r['LGBM_gap']:.2f} eV")
                st.markdown(
                    f"**Similarity to the training data:** the nearest training molecule has ECFP4 Tanimoto similarity "
                    f"**{r['similarity']:.2f}** (`{r['nearest_smiles']}`, DFT HOMO {r['nearest_HOMO']:.2f} / "
                    f"LUMO {r['nearest_LUMO']:.2f} eV). For test molecules this similar, the typical error (MAE) was "
                    f"**{r['expected_MAE_HOMO']:.2f} / {r['expected_MAE_LUMO']:.2f} eV** (HOMO / LUMO) for the GNN "
                    f"ensemble and {r['expected_LGBM_MAE_HOMO']:.2f} / {r['expected_LGBM_MAE_LUMO']:.2f} eV for LightGBM."
                )
                if 'dataset_HOMO' in r:
                    where = 'in the training set' if r['in_training_set'] else 'in the held-out test set'
                    st.info(f"This molecule is in CSD_EES ({r['dataset_structures']} crystal structure(s), {where}): "
                            f"DFT HOMO {r['dataset_HOMO']:.2f} eV, LUMO {r['dataset_LUMO']:.2f} eV.")
                for w in r['warnings']:
                    st.warning(w)
                with st.expander('Prediction of each model'):
                    st.dataframe(pd.DataFrame([{'model': m, 'HOMO (eV)': h, 'LUMO (eV)': l}
                                               for m, (h, l) in r['members'].items()]).round(3), hide_index=True)
            if show_atoms:
                with st.spinner('Computing atom contributions...'):
                    att = predictor.explain(r['mol'])
                st.caption('Each atom’s contribution to the prediction relative to a featureless graph of the same shape '
                           '(red raises, blue lowers the orbital energy; colours scaled per map).')
                a, b = st.columns(2)
                a.image(draw_attributions(r['mol'], att[:, 0]), caption='HOMO')
                b.image(draw_attributions(r['mol'], att[:, 1]), caption='LUMO')

with batch:
    st.markdown('One SMILES per line, optionally followed by a space and a name — or upload a CSV file with a '
                '`SMILES` column (and optionally a `name` column).')
    text = st.text_area('SMILES', height=180, placeholder='c1ccc2cc3ccccc3cc2c1 anthracene\nN#CC(C#N)=C1C=CC(=C(C#N)C#N)C=C1 TCNQ')
    upload = st.file_uploader('CSV file', type='csv')
    if st.button('Predict', type='primary'):
        if upload is not None:
            df = pd.read_csv(upload)
            col = next((c for c in df.columns if c.lower() == 'smiles'), None)
            if col is None:
                st.error('The CSV file needs a column called SMILES.')
                st.stop()
            smiles_list = df[col].astype(str).tolist()
            name_col = next((c for c in df.columns if c.lower() == 'name'), None)
            names = df[name_col].astype(str).tolist() if name_col else [''] * len(df)
        else:
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            smiles_list = [l.split(maxsplit=1)[0] for l in lines]
            names = [l.split(maxsplit=1)[1] if len(l.split(maxsplit=1)) > 1 else '' for l in lines]
        if not smiles_list:
            st.warning('Enter at least one SMILES.')
        else:
            bar = st.progress(0.0, text=f'Predicting {len(smiles_list)} molecules...')
            results = predictor.predict(smiles_list, progress=lambda f: bar.progress(f))
            bar.empty()
            table = predictor.to_table(results, names)
            n_bad = sum(r['error'] is not None for r in results)
            st.success(f'{len(results) - n_bad} predicted' + (f', {n_bad} could not be read' if n_bad else '') + '.')
            st.dataframe(table.round(3), hide_index=True)
            st.download_button('Download CSV', table.to_csv(index=False).encode(), 'homo_lumo_predictions.csv', 'text/csv')
