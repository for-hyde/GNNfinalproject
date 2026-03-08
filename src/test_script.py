# Just a testing script to test the functionality of the logger 
from utils.logging_utils import (start_log, log, log_section, log_warning, log_error)

start_log(log_dir="/workspace/logs/", run_name="test_run_1")

log_section("LOGGING")

log("This is a print statement")

log_warning("OH NO, THIS IS A WARNING")

log_error("THIS IS VERY BAD! START PANICKING")
