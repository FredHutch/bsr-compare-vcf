import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""## Compare VCFs""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # If the script is running in WASM (instead of local development mode), load micropip
    import sys
    if "pyodide" in sys.modules:
        import micropip
        running_in_wasm = True
    else:
        micropip = None
        running_in_wasm = False
    return micropip, running_in_wasm


@app.cell
async def _(micropip, mo, running_in_wasm):
    with mo.status.spinner("Loading dependencies"):
        # If we are running in WASM, some dependencies need to be set up appropriately.
        # This is really just aligning the needs of the app with the default library versions
        # that come when a marimo app loads in WASM.
        if running_in_wasm:
            print("Installing via micropip")
            # Downgrade plotly to avoid the use of narwhals
            await micropip.install("plotly<6.0.0")
            await micropip.install("ssl")
            micropip.uninstall("urllib3")
            micropip.uninstall("httpx")
            await micropip.install("urllib3==2.3.0")
            micropip.uninstall("requests")
            await micropip.install("requests==2.32.3")
            await micropip.install("httpx==0.26.0")
            await micropip.install("botocore==1.37.3")
            await micropip.install("jmespath==1.0.1")
            await micropip.install("s3transfer==0.11.3")
            await micropip.install("boto3==1.37.3")
            await micropip.install("aiobotocore==2.22.0")
            await micropip.install("cirro[pyodide]==1.5.4")  
            await micropip.install("anndata==0.11.3")  

        from typing import Dict, Optional, List, Tuple, Set
        from functools import lru_cache
        from collections import defaultdict
        from itertools import groupby
        from copy import copy

        from cirro import DataPortalLogin, DataPortalDataset
        from cirro.config import list_tenants
        from cirro.sdk.exceptions import DataPortalAssetNotFound
        from cirro.sdk.file import DataPortalFiles

        # A patch to the Cirro client library is applied when running in WASM
        if running_in_wasm:
            from cirro.helpers import pyodide_patch_all
            pyodide_patch_all()

    return DataPortalFiles, DataPortalLogin, List, list_tenants, lru_cache


@app.cell
def _(mo):
    # Get and set the query parameters
    query_params = mo.query_params()
    return (query_params,)


@app.cell
def _(list_tenants):
    # Get the tenants (organizations) available in Cirro
    tenants_by_name = {i["displayName"]: i for i in list_tenants()}
    tenants_by_domain = {i["domain"]: i for i in list_tenants()}

    def domain_to_name(domain):
        return tenants_by_domain.get(domain, {}).get("displayName")

    def name_to_domain(name):
        return tenants_by_name.get(name, {}).get("domain")
    return domain_to_name, tenants_by_name


@app.cell
def _(mo):
    mo.md(r"""## Connect to Database""")
    return


@app.cell
def _(mo):
    # Use a state element to manage the Cirro client object
    get_client, set_client = mo.state(None)
    return get_client, set_client


@app.cell
def _(domain_to_name, mo, query_params, tenants_by_name):
    # Let the user select which tenant to log in to (using displayName)
    domain_ui = mo.ui.dropdown(
        options=tenants_by_name,
        value=domain_to_name(query_params.get("domain")),
        label="Load Data from Cirro",
        on_change=lambda v: query_params.set("domain", v["domain"])
    )
    domain_ui
    return (domain_ui,)


@app.cell
def _(DataPortalLogin, domain_ui, get_client, mo):
    # If the user is not yet logged in, and a domain is selected, then give the user instructions for logging in
    # The configuration of this cell and the two below it serve the function of:
    #   1. Showing the user the login instructions if they have selected a Cirro domain
    #   2. Removing the login instructions as soon as they have completed the login flow
    if get_client() is None and domain_ui.value is not None:
        with mo.status.spinner("Authenticating"):
            # Use device code authorization to log in to Cirro
            cirro_login = DataPortalLogin(base_url=domain_ui.value["domain"])
            cirro_login_ui = mo.md(cirro_login.auth_message_markdown)
    else:
        cirro_login = None
        cirro_login_ui = None

    mo.stop(cirro_login is None)
    cirro_login_ui
    return (cirro_login,)


@app.cell
def _(cirro_login, set_client):
    # Once the user logs in, set the state for the client object
    set_client(cirro_login.await_completion())
    return


@app.cell
def _(get_client, mo):
    # Get the Cirro client object (but only take action if the user selected Cirro as the input)
    client = get_client()
    mo.stop(client is None)
    return (client,)


@app.cell
def _(get_client, mo):
    mo.stop(get_client() is not None)
    mo.md("*_Log in to view data_*")
    return


@app.cell
def _():
    # Helper functions for dealing with lists of objects that may be accessed by id or name
    def id_to_name(obj_list: list, id: str) -> str:
        if obj_list is not None:
            return {i.id: i.name for i in obj_list}.get(id)


    def name_to_id(obj_list: list) -> dict:
        if obj_list is not None:
            return {i.name: i.id for i in obj_list}
        else:
            return {}
    return id_to_name, name_to_id


@app.cell
def _(client):
    # Set the list of projects available to the user
    projects = client.list_projects()
    projects.sort(key=lambda i: i.name)
    return (projects,)


@app.cell
def _(id_to_name, mo, name_to_id, projects, query_params):
    # Let the user select which project to get data from
    project_ui = mo.ui.dropdown(
        label="Select Project:",
        value=id_to_name(projects, query_params.get("project")),
        options=name_to_id(projects),
        on_change=lambda v: query_params.set("project", v)
    )
    project_ui
    return (project_ui,)


@app.cell
def _(client, mo, project_ui):
    # Stop if the user has not selected a project
    mo.stop(project_ui.value is None)

    # Get the list of datasets available to the user
    datasets = client.get_project_by_id(project_ui.value).list_datasets()
    datasets.sort(key=lambda ds: ds.created_at, reverse=True)
    return (datasets,)


@app.cell
def _(datasets, id_to_name, mo, name_to_id, query_params):
    # Let the user select which dataset to get data from
    dataset_ui = mo.ui.multiselect(
        label="Select Datasets:",
        value=[id_to_name(datasets, query_dataset) for query_dataset in query_params.get("datasets", "").split(",") if id_to_name(datasets, query_dataset) is not None],
        options=name_to_id(datasets),
        on_change=lambda v: query_params.set("datasets", ','.join(v)),
        full_width=True
    )
    dataset_ui
    return (dataset_ui,)


@app.cell
def _(client, dataset_ui, mo, project_ui):
    # Stop if the user has not selected a dataset
    mo.stop(len(dataset_ui.value) == 0)

    # Get the selected datasets
    selected_datasets = [
        (
            client
            .get_project_by_id(project_ui.value)
            .get_dataset_by_id(dataset_uuid)
        )
        for dataset_uuid in dataset_ui.value
    ]

    mo.md("\n".join([f" - {ds.name} ({ds.process.name}) ({ds.created_at:%A, %B %d, %Y})" for ds in selected_datasets]))
    return (selected_datasets,)


@app.cell
def _(List, selected_datasets):
    # Get any VCF files from this dataset
    vcf_files: List[str] = [
        file.name[len("data/"):]
        for dataset in selected_datasets
        for file in dataset.list_files().filter_by_pattern("*.vcf.gz")
    ]
    return (vcf_files,)


@app.cell
def _(mo):
    with mo.status.spinner("Loading Dependencies"):
        import pandas as pd
    return (pd,)


@app.cell
def _(get_client, lru_cache, pd):
    @lru_cache
    def read_vcf(project_id: str, dataset_id: str, fp: str) -> pd.DataFrame:
        vcf = (
            get_client()
            .get_dataset(project=project_id, dataset=dataset_id)
            .list_files()
            .get_by_id("data/" + fp)
            .read_csv(
                sep="\t",
                comment='#',
                header=None,
                names=["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info", "format", "values"]
            )
        )

        # Parse out the extra values
        return pd.concat([
            vcf.drop(columns=["info", "format", "values"]),
            pd.DataFrame([
                dict(field.split("=") for field in r["info"].split(";") if len(field.split("=")) == 2)
                for _, r in vcf.iterrows()
            ], index=vcf.index),
            pd.DataFrame([
                dict(zip(
                    r['format'].split(":"),
                    r['values'].split(":")
                ))
                for _, r in vcf.iterrows()
            ], index=vcf.index)
        ], axis=1
        )
    return (read_vcf,)


@app.cell
def _(DataPortalFiles, mo):
    class ComparisonTool:
        name: str
        description: str

        def args1(self, vcf_files: DataPortalFiles):
            return mo.md("").batch()

        def args2(self, vcf_files: DataPortalFiles, **kwargs):
            return mo.md("").batch()

        def args3(self, vcf_files: DataPortalFiles, **kwargs):
            return mo.md("").batch()

        def display1(self, **kwargs):
            pass

        def display2(self, **kwargs):
            pass

        def display3(self, **kwargs):
            pass

    return (ComparisonTool,)


@app.cell
def _(mo):
    get_vcf1, set_vcf1 = mo.state(None)
    get_vcf2, set_vcf2 = mo.state(None)
    return get_vcf1, get_vcf2, set_vcf1, set_vcf2


@app.cell
def _(
    ComparisonTool,
    List,
    dataset_ui,
    get_vcf1,
    get_vcf2,
    mo,
    pd,
    project_ui,
    read_vcf,
    set_vcf1,
    set_vcf2,
):
    class CompareTwoVCFs(ComparisonTool):
        name = "Compare Two VCFs"
        description = "Compare the variants in two VCFs, taking the minor allele frequency (MAF) in to account."

        def args1(self, vcf_files: List[str]):
            # Let the user select the first VCF file
            return mo.md("{vcf1}").batch(
                vcf1=mo.ui.dropdown(
                    label="VCF 1",
                    options=vcf_files,
                    value=get_vcf1() if get_vcf1() in vcf_files else None,
                    on_change=set_vcf1
                )
            )

        def args2(self, vcf_files: List[str], vcf1: str):
            if vcf1 is None:
                return mo.md("Must specify a VCF file for comparison").batch()
            else:

                # Must choose a different VCF file
                return mo.md("{vcf2}").batch(
                    vcf2=mo.ui.dropdown(
                        label="VCF2",
                        options=[vcf for vcf in vcf_files if vcf != vcf1],
                        value=get_vcf2() if get_vcf2() in vcf_files and get_vcf2() != vcf1 else None,
                        on_change=set_vcf2
                    )
                )

        def display1(self, vcf1: str, vcf2: str):
            vcf1_df = read_vcf(project_ui.value, dataset_ui.value, vcf1)
            vcf2_df = read_vcf(project_ui.value, dataset_ui.value, vcf2)
            print(vcf1_df.head())
            print(vcf2_df.head())

            maf_df = pd.DataFrame({
                vcf1: vcf1_df.set_index("id")["MAF"].dropna(),
                vcf2: vcf2_df.set_index("id")["MAF"].dropna(),
            }).fillna(0)
            return maf_df
    return (CompareTwoVCFs,)


@app.cell
def _(CompareTwoVCFs, ComparisonTool, List):
    comparison_tools: List[ComparisonTool] = [
        CompareTwoVCFs
    ]
    return (comparison_tools,)


@app.cell
def _(comparison_tools: "List[ComparisonTool]", mo):
    # Let the user choose a comparison tool
    choose_tool = mo.ui.dropdown(
        label="Comparison Tool:",
        options=[t.name for t in comparison_tools],
        value=comparison_tools[0].name
    )
    choose_tool
    return (choose_tool,)


@app.cell
def _(choose_tool, comparison_tools: "List[ComparisonTool]"):
    tool = next(tool for tool in comparison_tools if tool.name == choose_tool.value)()
    return (tool,)


@app.cell
def _(tool, vcf_files: "List[str]"):
    args1 = tool.args1(vcf_files)
    args1
    return (args1,)


@app.cell
def _(args1, tool, vcf_files: "List[str]"):
    args2 = tool.args2(vcf_files, **args1.value)
    args2
    return (args2,)


@app.cell
def _(args1, args2, tool, vcf_files: "List[str]"):
    args3 = tool.args3(vcf_files, **args1.value, **args2.value)
    args3
    return (args3,)


@app.cell
def _(args1, args2, args3, tool):
    tool.display1(**args1.value, **args2.value, **args3.value)
    return


@app.cell
def _(args1, args2, args3, tool):
    tool.display2(**args1.value, **args2.value, **args3.value)
    return


@app.cell
def _(args1, args2, args3, tool):
    tool.display3(**args1.value, **args2.value, **args3.value)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
