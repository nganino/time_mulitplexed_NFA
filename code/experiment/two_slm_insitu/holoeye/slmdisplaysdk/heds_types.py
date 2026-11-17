# -*- coding: utf-8 -*-

#--------------------------------------------------------------------#
#                                                                    #
# Copyright (C) 2023 HOLOEYE Photonics AG. All rights reserved.      #
# Contact: https://holoeye.com/contact/                              #
#                                                                    #
# This file is part of HOLOEYE SLM Display SDK.                      #
#                                                                    #
# You may use this file under the terms and conditions of the        #
# "HOLOEYE SLM Display SDK Standard License v1.0" license agreement. #
#                                                                    #
#--------------------------------------------------------------------#


## \ingroup SLMDisplayPython
## The format of the SLM data.
class Format:

    ## No format was specified. Mainly used for default values.
    Undefined = 0

    ## A grayscale 8-bit data field.
    Grayscale8 = 1

    ## A grayscale float data field. The values must be between 0..1.
    GrayscaleFloat = 3

    ## A grayscale double data field. The values must be between 0..1.
    GrayscaleDouble = 4

    ## A 24bit data field with three color channels.
    RGB24 = 5

    ## A 32bit data field with four color channels.
    RGBA32 = 6


## \ingroup SLMDisplayPython
## A list of available load flags.
class LoadFlags:

    ## The default load behavior, optimized for performance.
    Default = 0

    ## Generate a colortable for the provided data. Does not apply to real data and phasevalues.
    CreateColorTable = 65536

    ## Shows the data after loading it and transfers the data right before showing it. Reduces wait time when showing data in real-time.
    ShowImmediately = 65536 * 2

    ## This load flag inverts the transpose option in showflags. Please use this when the API data is stored in Fortran memory layout instead of C memory layout.
    TransposeData = 65536# 4


## \ingroup SLMDisplayPython
## A list of available show flags.
class ShowFlags:

    ## Shows two-dimensional data unscaled at the center of the SLM. One-dimensional data is shown as a grating.
    PresentAutomatic = 0

    ## The data is shown unscaled at the center of the SLM.
    #  Free areas are filled with zeroes. The data is cropped to the SLM size.
    PresentCentered = 1

    ## If set, the data will fit the SLM so all data is visible and the aspect ratio is kept.
    #  Free areas on the top/bottom or left/right will be filled with zeroes.
    #  Only one of the present flags may be set.
    #  This option changes the scale of the displayed data. Therefore this show flag overwrites the transformScale option in a data handle.
    PresentFitWithBars = 2

    ## If set, the data will fit the SLM so the SLM is completely filled with data but the aspect ratio is kept.
    #  Some data might not be visible. Only one of the present flags may be set.
    #  This option changes the scale of the displayed data. Therefore this show flag overwrites the transformScale option in a data handle.
    PresentFitNoBars = 4

    ## If set, the data will completely fill the SLM. The aspect ratio will not be kept.
    #  In short the data is shown fullscreen. Only one of the present flags may be set.
    #  This option changes the scale of the displayed data. Therefore this show flag overwrites the transformScale option in a data handle.
    PresentFitScreen = 8

    ## Shows the given data in a tiling pattern. The pattern is tiled around the center of the SLM.
    PresentTiledCentered = 16

    ## If set, the rows and columns will be switched.
    TransposeData = 32

    ## If set, the data will be flipped horizontally.
    FlipHorizontal = 64

    ## If set, the data will be flipped vertically.
    FlipVertical = 128

    ## If set, the data will be inverted.
    InvertValues = 256


## \ingroup SLMDisplayPython
## The code for any error that occured.
class ErrorCode:

    ## No error.
    NoError = 0

    ## The given SLM instance does not exist.
    NoSLMConnected = 1

    ## The given filename is zero or too long.
    InvalidFilename = 2

    ## A filename was given, but the file does not exist.
    FileNotFound = 3

    ## A filename was given and the file exists, but the file format is not supported.
    UnsupportedFileFormat = 4

    ## The given data is zero or too long.
    InvalidDataPointer = 5

    ## The given data has a width less than one.
    InvalidDataWidth = 6

    ## The given data has a height less than one.
    InvalidDataHeight = 7

    ## A valid and supported filename or data was given, but the contained format is not supported.
    UnsupportedDataFormat = 8

    ## The renderer had an internal error, for example when updating the window or sprite.
    InternalRendererError = 9

    ## There is not enough system memory left to process the given filename or data.
    OutOfSystemMemory = 10

    ## Data transfer into video memory failed. There is either not enough video memory left on the GPU, or the maximum number of data handles with data on the GPU has been reached. Please release unused data handles and try again.
    OutOfVideoMemory = 11

    ## The current handle is invalid.
    InvalidHandle = 12

    ## The provided duration in frames is less than one or higher than 255.
    InvalidDurationInFrames = 13

    ## The given phase wrap must be greater than zero.
    InvalidPhaseWrap = 14

    ## Waiting for a datahandle to reach a certain state timed out and failed.
    WaitForHandleTimedOut = 15

    ## The number of Zernike values must be between zero and \ref ZernikeValues.COUNT.
    InvalidZernikeValueSize = 19

    ## The scale needs to be greater than zero.
    InvalidTransformScale = 20

    ## The value of a given enum is invalid.
    InvalidEnumValue = 21

    ## One of the arguments is invalid.
    InvalidArgument = 22

    ## The specified canvas does not exist.
    InvalidCanvas = 23

    ## The data is locked to another canvas.
    LockedToOtherCanvas = 24

    ## The specified custom shader could not be found.
    CustomShaderNotFound = 25

    ## The specified custom shader ha no data function.
    CustomShaderHasNoDataFunction = 26

    ## The custom shader could not be compiled.
    CustomerShaderFailedToCompile = 27

    ## Failed to connect to host.
    ConnectionFailed = 28

    ## Internal network timeout occurred.
    ConnectionTimedOut = 29

    ## There was an internal error during the connection.
    ConnectionInternalError = 30

    ## The handle does not belong to the given instance.
    HandleInstanceMismatch = 31

    ## The canvas does not belong to the given instance.
    CanvasInstanceMismatch = 32

    ## The given data was not a two-dimensional array.
    InvalidDataDimensions = 100

    ## The provided value for width_a is zero or less.
    GratingWidthAInvalid = 101

    ## The provided value for width_b is zero or less.
    GratingWidthBInvalid = 102

## \ingroup SLMDisplayPython
## The current state of the data.
class State:

    ## The data was just created.
    Issued = 0

    ## The given filename or data is waiting for processing.
    WaitingForProcessing = 1

    ## The given filename is being loaded.
    LoadingFile = 2

    ## The given or loaded data needs to be converted. Performance Warning!
    ConvertingData = 3

    ## The data is being processed for display. This is not about conversion but about processing the data.
    ProcessingData = 4

    ## The data is waiting to be transferred to the gpu.
    WaitingForTransfer = 5

    ## The data is uploaded to the gpu.
    TransferringData = 6

    ## The data is ready to be rendered. This is the end state of the loading process.
    ReadyToRender = 7

    ## The data is waiting to be rendered. This is the first state when showing data.
    WaitingForRendering = 8

    ## The data is being rendered. This is about the actual effort needed to render the data.
    Rendering = 9

    ## The data is waiting to become visible on the SLM. Only applies to deferred rendering.
    WaitingToBecomeVisible = 10

    ## The data is currently visible on the SLM.
    Visible = 11

    ## The data is currently visible on the SLM and the property durationInFrames has passed.
    VisibleDurationFinished = 12

    ## The data has been shown and is now no longer visible.
    Finished = 13

    ## An error occured. Check error code.
    Error = 14


## \ingroup SLMDisplayPython
## A list of the supported Zernike functions and their position in the list of values.
class ZernikeValues:

    ## The radius is always the first argument and is required. It is provided in pixels.
    RadiusPx = 0

    ## The Tilt X function. f = x
    TiltX = 1

    ## The Tilt Y function. f = y
    TiltY = 2

    ## The Astig 45deg function. f = 2xy
    Astig45 = 3

    ## The Defocus function. f = 1-2x^2-2y^2
    Defocus = 4

    ## The Astig 0deg function. f = y^2-x^2
    Astig0 = 5

    ## The Trifoil X function. f = 3xy^2-x^3
    TrifoilX = 6

    ## The Coma X function. f = -2x+3xy^2+3x^3
    ComaX = 7

    ## The Coma Y function. f = -2y+3y^3+3x^2y
    ComaY = 8

    ## The Trifoil Y function. f = y^3-3x^2y
    TrifoilY = 9

    ## The Quadrafoil X function. f = 4y^3x-4x^3y
    QuadrafoilX = 10

    ## The Astig 2nd 45deg function. f = -6xy+8y^3x+8x^3y
    Astig2nd45 = 11

    ## The Spherical ABB function. f = 1-6y^2-6x^2+6y^4+12x^2y^2+6x^4
    SphericalABB = 12

    ## The Astig 2nd 0deg function. f = -3y^2+3x^2+4y^4-4x^2y^2-4x^4
    Astig2nd0 = 13

    ## The Quadrafoil Y function. f = y^4-6x^2y^2+x^4
    QuadrafoilY = 14

    ## The number of supported Zernike values.
    COUNT = 15


## \ingroup SLMDisplayPython
## Represents the different datahandle values which can be applied/written during an update.
class ApplyDataHandleValue:

    ## No values will be applied.
    ApplyNone = 0

    ## The value of \ref Datahandle.durationInFrames will be applied.
    DurationInFrames = 1

    ## The value of \ref Datahandle.dataFlags will be applied.
    ShowFlags = 2

    ## The value of \ref Datahandle.phaseWrap will be applied.
    PhaseWrap = 4

    ## The value of \ref Datahandle.transformShiftX, \ref Datahandle.transformShiftY and \ref Datahandle.transformScale will be applied.
    Transform = 8

    ## The value of \ref Datahandle.beamSteerX, \ref Datahandle.beamSteerY and \ref Datahandle.beamLens will be applied.
    BeamManipulation = 16

    ## The value of \ref Datahandle.gamma will be applied.
    Gamma = 32

    ## The value of \ref Datahandle.valueOffset will be applied.
    ValueOffset = 64


## \ingroup SLMDisplayPython
## Represents the different settings for the preview window.
class SLMPreviewFlags:

    ## No settings will be applied.
    NoFlag = 0

    ## Disables the border of the preview window.
    NoBorder = 1

    ## Makes sure the window is on top of other windows.
    OnTop = 2

    ## Shows the Zernike radius.
    ShowZernikeRadius = 4

    ## Show the wavefront compensation in the preview.
    ShowWavefrontCompensation = 8


## \ingroup SLMDisplayPython
## Represents the different settings for the wavefront compensation. \see WavefrontcompensationFlags
class WavefrontcompensationFlags:

    ## No flags will be applied.
    NoFlag = 0

    ## Flips the wavefront compensation horizontally.
    FlipHorizontally = 1

    ## Flips the wavefront compensation vertically.
    FlipVertically = 2



