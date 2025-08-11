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
        from io import StringIO

        from cirro import DataPortalLogin, DataPortalDataset
        from cirro.config import list_tenants
        from cirro.sdk.exceptions import DataPortalAssetNotFound
        from cirro.sdk.file import DataPortalFiles, DataPortalFile

        # A patch to the Cirro client library is applied when running in WASM
        if running_in_wasm:
            from cirro.helpers import pyodide_patch_all
            pyodide_patch_all()

    return (
        DataPortalDataset,
        DataPortalFile,
        DataPortalFiles,
        DataPortalLogin,
        Dict,
        List,
        StringIO,
        Tuple,
        list_tenants,
        lru_cache,
    )


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

    # Make sure that every dataset has a different name
    if len(set([ds.name for ds in selected_datasets])) < len(selected_datasets):
        _msg = "Error: dataset names must be unique"
    else:
        _msg = "\n".join([f" - {ds.name} ({ds.process.name}) ({ds.created_at:%A, %B %d, %Y})" for ds in selected_datasets])

    dataset_name_dict = {
        ds.name: ds.id for ds in selected_datasets
    }

    mo.md(_msg)
    return dataset_name_dict, selected_datasets


@app.cell
def _(DataPortalDataset, DataPortalFile, Tuple):
    def vcf_file_to_name(dataset: DataPortalDataset, file: DataPortalFile) -> str:
        return dataset.name + "/" + file.name[len("data/"):] + f" ({file.size})"


    def vcf_name_to_file(name_str: str) -> Tuple[str, str]:
        return name_str.rsplit(" (", 1)[0].split("/", 1)
    return vcf_file_to_name, vcf_name_to_file


@app.cell
def _(List, selected_datasets, vcf_file_to_name):
    # Get any VCF files from this dataset
    vcf_files: List[str] = [
        vcf_file_to_name(dataset, file)
        for dataset in selected_datasets
        for file in dataset.list_files().filter_by_pattern("*.vcf.gz")
    ]
    return (vcf_files,)


@app.cell
def _(mo):
    with mo.status.spinner("Loading Dependencies"):
        import pandas as pd
    return (pd,)


@app.function
def find_header(content: str):
    for line in content.split("\n"):
        if line.startswith("#CHROM"):
            return line[1:].rstrip("\n").split("\t")


@app.cell
def _(Dict, StringIO, get_client, lru_cache, mo, pd):
    @lru_cache
    def read_vcf_cache(project_id: str, dataset_id: str, fp: str) -> Dict[str, pd.DataFrame]:
        """
        Return a dict with one element for each of the samples defined in a VCF.
        """
        fixed_header = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        with mo.status.spinner(f"Reading VCF ({fp})"):
            content = (
                get_client()
                .get_dataset(project=project_id, dataset=dataset_id)
                .list_files()
                .get_by_id("data/" + fp)
                .read(compression="gzip")
            )

            # Find the header
            header = find_header(content)

            # Make sure the first few fields are the same
            for i, cname in enumerate(fixed_header):
                assert header[i] == cname, f"Expected header field {i+1} to be {cname} (found {', '.join(header)})"

            vcf = (
                pd.read_csv(
                    StringIO(content),
                    sep="\t",
                    comment='#',
                    header=None,
                    names=header
                )
            )

            # Make a separate DataFrame for each of the samples in the VCF
            return {
                cname: parse_vcf_sample(cvals)
                for cname, cvals in vcf.set_index(fixed_header).items()
            }

    def parse_vcf_sample(cvals):
        return pd.DataFrame(
            [
                dict(zip(
                    format_fields.split(":"),
                    value_fields.split(":")
                ))
                for format_fields, value_fields in zip(cvals.reset_index()["FORMAT"].values, cvals.values)
            ],
            index=cvals.index
        ).reset_index()

    return (read_vcf_cache,)


@app.cell
def _(
    Dict,
    dataset_name_dict,
    pd,
    project_ui,
    read_vcf_cache,
    vcf_name_to_file,
):
    def read_vcf(fp: str) -> Dict[str, pd.DataFrame]:
        """
        Return a dict with one element for each of the samples defined in a VCF.
        """
        # Get the dataset name from the selected fp
        ds_name, vcf_fp = vcf_name_to_file(fp)
        # Get the dataset ID
        ds_id = dataset_name_dict[ds_name]

        # Call the cached function
        vcf_dict = read_vcf_cache(project_ui.value, ds_id, vcf_fp)

        # If a sample was specified
        if "):" in fp:
            samplename = fp.rsplit("):", 1)[1]
            return vcf_dict[samplename]
        else:
            return vcf_dict
    return (read_vcf,)


@app.cell
def _(mo):
    with mo.status.spinner("Loading dependencies"):
        import plotly.express as px
        import plotly.graph_objects as go
        import math
        import numpy as np

    return go, math, np, px


@app.cell
def _(DataPortalFiles, Dict, List, mo, pd):
    class ComparisonTool:
        """
        Generic base class which can be used to build comparison tools.
        Each of the args functions can be used sequentially to collect information from the user.
        Each of the display functions will be presented after the arguments have all been collected.
        """
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

        def read_and_combine_vcfs(selected_vcfs: List[str]) -> Dict[str, pd.DataFrame]:
            """
            When the user selects 
            """

    return (ComparisonTool,)


@app.cell
def _(mo):
    # State elements are used to maintain the same user selections when the argument cells are rerun
    get_selected_vcfs, set_selected_vcfs = mo.state([])
    get_vcf1, set_vcf1 = mo.state(None)
    get_vcf2, set_vcf2 = mo.state(None)
    get_opacity, set_opacity = mo.state(0.5)
    get_af_threshold, set_af_threshold = mo.state(0.25)
    get_figure_width, set_figure_width = mo.state(600)
    return (
        get_af_threshold,
        get_figure_width,
        get_opacity,
        get_selected_vcfs,
        get_vcf1,
        get_vcf2,
        set_af_threshold,
        set_figure_width,
        set_opacity,
        set_selected_vcfs,
        set_vcf1,
        set_vcf2,
    )


@app.cell
def _(
    ComparisonTool,
    List,
    get_af_threshold,
    get_figure_width,
    get_opacity,
    get_selected_vcfs,
    get_vcf1,
    get_vcf2,
    go,
    math,
    mo,
    np,
    pd,
    px,
    read_vcf,
    set_af_threshold,
    set_figure_width,
    set_opacity,
    set_selected_vcfs,
    set_vcf1,
    set_vcf2,
):
    class CompareTwoSamples(ComparisonTool):
        name = "Compare Two Samples"
        description = """
        Compare the variants in two samples, taking the minor allele frequency (MAF) in to account.
        The samples may be recorded in a single VCF file or across multiple VCF files.
        """

        def args1(self, vcf_files: List[str]):
            # Let the user select one or more VCF files
            return mo.md(" - {selected_vcfs}").batch(
                selected_vcfs=mo.ui.multiselect(
                    label="VCF File(s)",
                    options=vcf_files,
                    value=[f for f in get_selected_vcfs() if f in vcf_files],
                    on_change=set_selected_vcfs
                )
            )

        def args2(self, vcf_files: List[str], selected_vcfs: List[str]):
            if len(selected_vcfs) == 0:
                return mo.md("Must specify one or more VCF files for comparison").batch()
            else:

                # Read in the samples encoded in those VCF file(s)
                # Encode the keys as filename:samplename
                vcf_data = {
                    f"{file_uri}:{samplename}": vcf_df
                    for file_uri in selected_vcfs
                    for samplename, vcf_df in read_vcf(file_uri).items()
                }
                options = list(vcf_data.keys())
                options.sort()

                # Select the first VCF sample
                return mo.md(" - {vcf1}").batch(
                    vcf1=mo.ui.dropdown(
                        label="VCF1",
                        options=options,
                        value=(
                            get_vcf1()
                            if get_vcf1() in options
                            else (
                                options[0] if len(options) > 0
                                else None
                            )
                        ),
                        on_change=set_vcf1
                    )
                )

        def args3(self, vcf_files: List[str], selected_vcfs: List[str], vcf1: str):

            if len(selected_vcfs) == 0:
                return mo.md("Must specify one or more VCF files for comparison").batch()
            else:

                # Read in the samples encoded in those VCF file(s)
                # Encode the keys as filename:samplename
                vcf_data = {
                    f"{file_uri}:{samplename}": vcf_df
                    for file_uri in selected_vcfs
                    for samplename, vcf_df in read_vcf(file_uri).items()
                    if f"{file_uri}:{samplename}" != vcf1
                }
                options = list(vcf_data.keys())
                options.sort()

                # Select which samples to display
                return mo.md("""
                - {vcf2}
                - {af_threshold}
                - {opacity}
                - {width}
                """).batch(
                    vcf2=mo.ui.dropdown(
                        label="VCF2",
                        options=options,
                        value=(
                            get_vcf2()
                            if get_vcf2() in options
                            else (
                                options[0]
                                if len(options) > 0
                                else None
                            )
                        ),
                        on_change=set_vcf2
                    ),
                    af_threshold=mo.ui.number(
                        label="Allele Frequency Threshold",
                        start=0.,
                        stop=1.,
                        value=get_af_threshold(),
                        on_change=set_af_threshold
                    ),
                    opacity=mo.ui.number(
                        label="Point Opacity:",
                        start=0.,
                        stop=1.,
                        value=get_opacity(),
                        on_change=set_opacity
                    ),
                    width=mo.ui.number(
                        label="Figure Width:",
                        start=100,
                        value=get_figure_width(),
                        on_change=set_figure_width
                    )
                )

        def display1(self, selected_vcfs: List[str], vcf1: str, vcf2: str, af_threshold: float, opacity: float, width: int):

            msg = ""
            if vcf1 is None:
                msg += "\n - Must specify a VCF1 file for comparison"
            else:
                vcf1_df = read_vcf(vcf1)
                if "AF" not in vcf1_df:
                    msg += f"\n - AF field missing from {vcf1}"

            if vcf2 is None:
                msg += "\n - Must specify a VCF2 file for comparison"
            else:
                vcf2_df = read_vcf(vcf2)
                if "AF" not in vcf2_df:
                    msg += f"\n - AF field missing from {vcf2}"

            if len(msg) > 0:
                return mo.md(msg)

            with mo.status.spinner("Merging AFs:"):
                af_df = (
                    pd.DataFrame({
                        vcf1: vcf1_df.set_index(["CHROM", "POS", "REF", "ALT"])["AF"].dropna(),
                        vcf2: vcf2_df.set_index(["CHROM", "POS", "REF", "ALT"])["AF"].dropna(),
                    })
                    .fillna(0)
                    .astype(float)
                    .reset_index()
                )

            return mo.vstack([
                self.scatter(af_df, vcf1, vcf2, opacity, af_threshold, width),
                self.venn(af_df, vcf1, vcf2, af_threshold, width)
            ])

        def scatter(self, af_df: pd.DataFrame, vcf1: str, vcf2: str, opacity: float, af_threshold: float, width: int):

            fig = px.scatter(
                data_frame=af_df,
                x=vcf1,
                y=vcf2,
                hover_data=["CHROM", "POS", "REF", "ALT"],
                # Do not include the filename in the display
                labels={
                    vcf1: vcf1.split("):")[-1],
                    vcf2: vcf2.split("):")[-1]
                },
                template="simple_white",
                opacity=opacity,
                width=width
            )
            fig.add_vline(x=af_threshold, line_dash="dash", line_width=3)
            fig.add_hline(y=af_threshold, line_dash="dash", line_width=3)
            return mo.ui.plotly(fig)

        def venn(self, af_df: pd.DataFrame, vcf1: str, vcf2: str, af_threshold: float, width: int):

            # Get the number of total variants passing the threshold for either sample, and the number shared
            totals = {cname: (cvals >= af_threshold).sum() for cname, cvals in af_df.items() if cname in [vcf1, vcf2]}
            shared = (
                af_df
                .query(f"`{vcf1}` >= {af_threshold}")
                .query(f"`{vcf2}` >= {af_threshold}")
                .shape[0]
            )

            # Make a figure
            fig = go.Figure()

            # Make a circle for each sample
            venn_coords, xrange, yrange, overlap_x = self.venn_coords(totals[vcf1], totals[vcf2], shared)
            for sample_name, coords in zip([vcf1, vcf2], venn_coords):
                fig.add_shape(
                    type="circle",
                    line_width=5,
                    xref="x",
                    yref="y",
                    fillcolor="Blue",
                    line_color="DarkBlue",
                    **coords
                )

                # Add a text label
                fig.add_annotation(
                    text=f'{sample_name.rsplit("):", 1)[1]}={totals[sample_name]:,}',
                    x=np.mean([coords["x0"], coords["x1"]]),
                    y=0,
                    showarrow=False
                )

            # Annotate the overlap
            fig.add_annotation(
                text=f"Shared={shared:,}",
                x=overlap_x,
                y=yrange[1] * -0.1,
                ax=0,
                ay=yrange[1],
                showarrow=True,
                arrowwidth=1,
                arrowhead=5,
                arrowcolor="black",
            )

            fig.update_xaxes(range=xrange, zeroline=False, visible=False)
            fig.update_yaxes(range=yrange, zeroline=False, visible=False)

            # Get the x and y span so that the figure size can be computed to be square
            xspan = xrange[1] - xrange[0]
            yspan = yrange[1] - yrange[0]
            height = (width / xspan) * yspan

            fig.update_layout(template="simple_white", width=width, height=height)

            return mo.ui.plotly(fig)

        def venn_coords(self, total_a: int, total_b: int, shared: int):
            """Calculate the venn coordinates showing overlap between two sets."""

            # Calculate the radius of the circles which give the needed areas
            radius_a = math.sqrt(total_a / math.pi)
            radius_b = math.sqrt(total_b / math.pi)

            # The circle for A will be centered on 0/0
            # Find the position of B where the overlapping space is closest to the actual
            calc_shared = {
                x: self.calculate_overlap_area(0, 0, radius_a, x, 0, radius_b)
                for x in np.arange(0, 1.2 * (radius_a + radius_b), (radius_a + radius_b) / 1000)
            }
            # Find the difference between the computed overlap and the desired
            error = {x: np.abs(shared - a) for x, a in calc_shared.items()}

            # Get the smallest value of X that has the lowest error
            x_b = next(x for x, e in error.items() if e <= np.min(list(error.values())))

            # Compute the range of each axis to display
            xrange = (1.1 * min(-radius_a, x_b - radius_b), 1.1 * max(radius_a, x_b + radius_b))
            yrange = (1.1 * min(-radius_a, -radius_b), 1.1 * max(radius_a, radius_b))

            # Compute the middle of the x overlap for the circles
            overlap_x = np.mean([radius_a, x_b - radius_b])

            # Return the x/y coordinates for those two circles
            return [
                dict(
                    x0=-radius_a,
                    y0=-radius_a,
                    x1=radius_a,
                    y1=radius_a
                ),
                dict(
                    x0=x_b - radius_b,
                    y0=-radius_b,
                    x1=x_b + radius_b,
                    y1=radius_b
                )
            ], xrange, yrange, overlap_x

        @staticmethod
        def calculate_overlap_area(x1, y1, r1, x2, y2, r2):
            """
            Calculates the area of overlap between two circles.

            Args:
                x1, y1, r1: Center coordinates and radius of the first circle.
                x2, y2, r2: Center coordinates and radius of the second circle.

            Returns:
                The area of overlap.
            """
            d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # No overlap
            if d >= r1 + r2:
                return 0.0
            # One circle inside another
            elif d <= abs(r1 - r2):
                return math.pi * min(r1, r2)**2
            # Partial overlap
            else:
                # Angles for the circular segments
                alpha1 = math.acos((d**2 + r1**2 - r2**2) / (2 * d * r1))
                alpha2 = math.acos((d**2 + r2**2 - r1**2) / (2 * d * r2))

                # Area of circular segments
                area1 = r1**2 * alpha1 - 0.5 * r1**2 * math.sin(2 * alpha1)
                area2 = r2**2 * alpha2 - 0.5 * r2**2 * math.sin(2 * alpha2)

                return area1 + area2

    return (CompareTwoSamples,)


@app.cell
def _(CompareTwoSamples, ComparisonTool, List):
    comparison_tools: List[ComparisonTool] = [
        CompareTwoSamples
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
