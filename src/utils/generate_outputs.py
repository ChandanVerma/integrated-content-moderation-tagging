def output_template(kinesis_event, new_video, message_receive_time):
    """Generates an output dictionary to be sent to SQS and \
    eventually a Snowflake table.

    Args:
        kinesis_event (dict): event payload
        new_video (str): url to lomotif
        message_receive_time (string): UTC time when the message is \
        received by the endpoint.

    Returns:
        _type_: _description_
    """
    default_time = "0000-00-00 00:00:00.000000+00:00"
    # general outputs
    output_dict = {}
    output_dict["LOMOTIF_ID"] = kinesis_event["lomotif"]["id"]
    output_dict["VIDEO"] = new_video
    output_dict["COUNTRY"] = kinesis_event["lomotif"]["country"]
    output_dict["CREATION_TIME"] = kinesis_event["lomotif"]["created"]
    output_dict["MESSAGE_RECEIVE_TIME"] = message_receive_time
    output_dict["KEY_FRAMES"] = ""
    output_dict["NUM_FRAMES"] = ""
    output_dict["FPS"] = ""

    # nudenet outputs
    output_dict["NN_PROCESS_START_TIME"] = default_time
    output_dict["NN_PREDICTION_TIME"] = default_time
    output_dict["NN_SAFE_SCORES"] = ""
    output_dict["NN_UNSAFE_SCORES"] = ""
    output_dict["NN_TO_BE_MODERATED"] = True
    output_dict["NN_PREDICTION_SUCCESS"] = False
    output_dict["NN_STATUS"] = -1

    # clip outputs
    output_dict["CLIP_PROCESS_START_TIME"] = default_time
    output_dict["CLIP_PREDICTION_TIME"] = default_time
    output_dict["CLIP_PREDICTION_SUCCESS"] = False
    output_dict["CLIP_TO_BE_MODERATED"] = True
    output_dict["CLIP_STATUS"] = -1

    # coop outputs
    output_dict["COOP_PROCESS_START_TIME"] = default_time
    output_dict["COOP_PREDICTION_TIME"] = default_time
    output_dict["COOP_PREDICTION_SUCCESS"] = False
    output_dict["COOP_STATUS"] = -1

    # coop-clip outputs
    output_dict["PREDICTED_PRIMARY_CATEGORY"] = ""
    output_dict["PREDICTED_SECONDARY_CATEGORY"] = ""
    output_dict["PREDICTED_TOP3_PRIMARY_CATEGORY"] = ""
    output_dict["PREDICTED_TOP3_SECONDARY_CATEGORY"] = ""

    # middle finger outputs
    output_dict["MFD_PROCESS_START_TIME"] = default_time
    output_dict["MFD_PREDICTION_TIME"] = default_time
    output_dict["MFD_TO_BE_MODERATED"] = True
    output_dict["MFD_PREDICTION_SUCCESS"] = False
    output_dict["MFD_STATUS"] = -1

    # other attributes
    output_dict["MODEL_VERSION"] = "2.3.0"
    return output_dict
