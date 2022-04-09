import logging
import os

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("consoleLogger")


def return_processing_function(file):
    """
    Determines how the abstracts are presented, and selects the appropriate processing function.

    Parameters
    ----------
    file : str
        Either a directory, .txt or .csv file

    Returns
    -------
    processing_function : title generator function
        a function used for generating titles
    """
    if file.endswith(".txt"):
        processing_function = generate_title
    elif file.endswith(".csv"):
        processing_function = batch_processing
    elif os.path.isdir(file):
        processing_function = batch_processing
    else:
        logger.error(
            "Filetype not recognised, please supply a directory, a '.txt' or a '.csv' file."
        )
        raise (ValueError)

    return processing_function


@click.command()
@click.option(
    "--file", help="Target file or directory for generating titles", default=None
)
@click.option("--abstract", help="Abstract to process", default=None)
@click.option("--category", help="Category of the abstract", default=None)
def main(file, abstract, category):
    """
    Generate a title, or multiple titles, per supplied abstract and category combination.
    The 'file' parameter can take multiple values:
        .txt :
            Generate a title for a single abstract
        .csv :
            Generate a title for every abstract in the csv.
        directory :
            Generate a title for every .txt in a directory.
    
    Alternatively, one may directly supply an abstract and a category for which to generate a title.
    When directly supplied, these override file. 

    Parameters
    ----------
    file : str
        Which file or directory to process.
    abstract : str
        Directly supply an abstract for processing.
    category : str
        Directly supply an category.
    """
    if file is None and abstract is None:
        logger.error(
            "Either supply a file/directory to process, or directly supply an abstract and category."
        )
        raise (ValueError)
    elif abstract is not None:
        processing_function = generate_title
    elif file is not None:
        processing_function = return_processing_function(file)
    processing_function(data)


if __name__ == "__main__":
    main()
