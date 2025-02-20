# Initially intended to store function mappings in a config json, but it is much easier to allocate dynamic functions to commands in a dictionary.
import utilfunctions as uf

mapping = {
    "HELP" : {
        "function" : uf.get_help_message,
        "description" : "Display help message."
    }
}