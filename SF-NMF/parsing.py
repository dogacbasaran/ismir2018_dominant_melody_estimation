import optparse


def parseOptions(argsin,wavfilerequired = False):

    usage = "usage: %prog [options] inputAudioFile"
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage)
    # Name of the output files:
    parser.add_option("-i", "--input-file",
                      dest="input_file", type="string",
                      help="Path of the input file.\n",
                      default=None)
    parser.add_option("-o", "--pitch-output",
                      dest="pitch_output_file", type="string",
                      help="name of the output file for an external algorithm.\n"
                           "If None appends _pitches to the wav",
                      default=None)
    parser.add_option("-s", "--pitch-salience-output-file",
                      dest="sal_output_file", type="string",
                      help="name of the output file for the Salience File.\n"
                           "If None the salience file is not saved.",
                      default=None)

    parser.add_option("-v", "--vit-pitch-output-file",
                      dest="vit_pitch_output_file", type="string",
                      help="name of the output file for the estimated pitches with Viterbi.\n"
                           "If None it does not execute the Viterbi extraction",
                      default=None)

    parser.add_option("-p", "--pitch-output-file",
                      dest="pitch_output_file", type="string",
                      help="name of the output file for an external algorithm.\n"
                           "If None appends _pitches to the wav",
                      default=None)
    # Some more optional options:
    parser.add_option("-d", "--with-display", dest="displayEvolution",
                      action="store_true",help="display the figures",
                      default=False)
    parser.add_option("-q", "--quiet", dest="verbose",
                      action="store_false",
                      help="use to quiet all output verbose",
                      default=False)
    parser.add_option("--nb-iterations", dest="nbiter",
                      help="number of iterations", type="int",
                      default=40)

    parser.add_option("--expandHF0Val", dest="expandHF0Val",
                      help="value for expanding the distribution of the values of HF0", type="float",
                      default=1)

    parser.add_option("--window-size", dest="windowSize", type="float",
                      default=0.04644,help="size of analysis windows, in s.")
    parser.add_option("--Fourier-size", dest="fourierSize", type="int",
                      default=None,
                      help="size of Fourier transforms, "\
                           "in samples.")
    parser.add_option("--hopsize", dest="hopsize", type="float",
                      default=0.01,
                      help="size of the hop between analysis windows, in s.")
    parser.add_option("--nb-accElements", dest="R", type="float",
                      default=40.0,
                      help="number of elements for the accompaniment.")
    parser.add_option("--WM-start", dest="WMstart", type="int",
                      default=-1,
                      help="number of iterations before start updating WM")
    parser.add_option("--numAtomFilters", dest="P_numAtomFilters",
                      type="int", default=30,
                      help="Number of atomic filters - in WGAMMA.")
    parser.add_option("--numFilters", dest="K_numFilters", type="int",
                      default=10,
                      help="Number of filters for decomposition - in WPHI")
    parser.add_option("--min-F0-Freq", dest="minF0", type="float",
                      default=55.0,
                      help="Minimum of fundamental frequency F0.")
    parser.add_option("--max-F0-Freq", dest="maxF0", type="float",
                      default=1760.0,
                      help="Maximum of fundamental frequency F0.")
    parser.add_option("--samplingRate", dest="Fs", type="float",
                      default=22050,
                      help="Sampling rate")
    parser.add_option("--step-F0s", dest="stepNotes", type="int",
                      default=5,
                      help="Number of F0s in dictionary for each semitone.")
    # PitchContoursMelody
    parser.add_option("--voicingTolerance", dest="voicingTolerance", type="float",
                      default=0.2,
                      help="Allowed deviation below the average contour mean salience of all contours (fraction of the standard deviation)")

    #PitchContours
    parser.add_option("--peakDistributionThreshold", dest="peakDistributionThreshold", type="float",
                      default=0.9,
                      help="Allowed deviation below the peak salience mean over all frames (fraction of the standard deviation)")

    parser.add_option("--peakFrameThreshold", dest="peakFrameThreshold", type="float",
                      default=0.9,
                      help="Per-frame salience threshold factor (fraction of the highest peak salience in a frame)")

    parser.add_option("--minDuration", dest="minDuration", type="float",
                      default=100,
                      help="the minimum allowed contour duration [ms]")

    parser.add_option("--timeContinuity", dest="timeContinuity", type="float",
                      default=100,
                      help="Time continuity cue (the maximum allowed gap duration for a pitch contour) [ms]")
    parser.add_option("--voiceVibrato",dest = "voiceVibrato",default =False, help="detect voice vibrato for melody estimation")

    parser.add_option("--pitchContinuity", dest="pitchContinuity", type="float",
                      default=27.5625,
                      help="pitch continuity cue (maximum allowed pitch change durig 1 ms time period) [cents]")

    parser.add_option("--extractionMethod", dest="extractionMethod", type="string",
                      help="name of the method to be executed, if None, default is BG2, with PCS (Pitch Contour Selection)",
                      default="BG2")

    parser.add_option("--background-basis", dest="background_basis", type="string",
                      help="Background basis vector input: oracle/cluster/standard",
                      default="standard")

    parser.add_option("--hgamma-basis", dest="hgamma_basis", type="string",
                      help="HGAMMA basis vector input: oracle/cluster/standard",
                      default="standard")

    parser.add_option("--alpha", dest="alpha", type="float",
                      default=0.0,
                      help="Sparsity constraint coefficient for HF0")

    parser.add_option("--beta", dest="beta", type="float",
                      default=0.0,
                      help="Smoothness constraint coefficient for HPHI")

    parser.add_option("--smooth-HPHI", dest="smooth_HPHI",
                      action="store_true",help="Apply smoothness constraint on HPHI",
                      default=False)

    parser.add_option("--sparse-HF0", dest="sparse_HF0",
                      action="store_true",help="Apply sparsity constraint on HF0",
                      default=False)

    parser.add_option("--pitch-corrected", dest="pitch_corrected",
                      action="store_true", help="Tune the basis functions for the tuning of the song",
                      default=False)

    (options, args) = parser.parse_args(argsin)
    # if the argument is not given with -i
    if len(args)>0:
        options.input_file = args[0]
    options.hopsizeInSamples = int(round(options.hopsize*options.Fs))

    if ((len(args) < 1) & wavfilerequired):
        parser.error("incorrect number of arguments, use option -h for help.")

    return args, options
