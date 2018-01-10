versastat
=========

python interface for VersaSTAT automation. See `examples/` for demo/test code.


versastat.control
-----------------
Instruments can be controlled via the VersaSTATControl library.
From the SDK docs, it seems that a series of experiments can be set up (argument lists are passed as comma-separated strings...) and run asynchronously.
It should be possible to live-stream the data from the instrument via calls to Experiment.GetData* to build a real-time (interactive?) UI.


versastat.position
------------------
Motion controller: XCD controller by nanomotion.
Appears to be https://www.nanomotion.com/wp-content/uploads/2014/05/XCD-controller-user-manual.pdf
