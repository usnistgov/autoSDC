versastat

python interface for VersaSTAT automation.

Instruments can be controlled via the VersaSTATControl library.
From the SDK docs, it seems that a series of experiments can be set up (argument lists are passed as comma-separated strings...) and run asynchronously.
It should be possible to live-stream the data from the instrument via calls to Experiment.GetData* to build a real-time (interactive?) UI.

