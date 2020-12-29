import logging

from google.protobuf.message import Message
from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer)
from google.protobuf.pyext._message import RepeatedCompositeContainer


def set_oneof_field(fieldname, msg, val):
    """Sets a 'oneof' field inside of 'msg'
    Parameters
    ----------
    fieldname: str
        the oneof name of the field
    msg: google.protobuf.Message
        the protobuf to modify
    val: google.protobuf.Message, numeric, string
        the value to set

    Example
    -------
    > protomsg = pybmix.proto.hierarchy_prior_pb2.NNIGPrior()
    > prior = pybmix.proto.hierarchy_prior_pb2.NNIGPrior.FixedValues(mean=1)
    > set_oneof_field("prior", protomsg, prior)
    """
    if fieldname not in msg.DESCRIPTOR.oneofs_by_name.keys():
        logging.error("fieldname {0} is not a 'oneof' field of {1}".format(
            fieldname, msg.DESCRIPTOR.name))
        return

    onoef_names = get_oneof_names(fieldname, msg)
    for name in onoef_names:
        try:
            success = set_shallow_field(name, msg, val)
            if success:
                return True

        except Exception as e:
            continue

    return False


def get_oneof_names(oneof_name, msg):
    return [x.name for x in msg.DESCRIPTOR.oneofs_by_name[oneof_name].fields]


def get_oneof_types(oneof_name, msg):
    return [x.message_type.name for x in 
            msg.DESCRIPTOR.oneofs_by_name[oneof_name].fields]


def set_shallow_field(fieldname, msg, val):
    typ = type(getattr(msg, fieldname))
    success = False
    if typ == RepeatedScalarFieldContainer:
        rep = getattr(msg, fieldname)
        rep.append(val)
        success = True
    elif typ in {RepeatedCompositeFieldContainer,
                RepeatedCompositeContainer}:
        rep = getattr(msg, fieldname)
        rep.add().CopyFrom(val)
        success = True
    else:
        if isinstance(getattr(msg, fieldname), Message):
            rep = getattr(msg, fieldname)
            rep.CopyFrom(val)
            success = True
        else:
            setattr(msg, fieldname, val)
            success = True
    
    return success


