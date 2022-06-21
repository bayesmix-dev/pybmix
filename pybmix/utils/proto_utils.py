import logging

from google.protobuf.internal.containers import (
    MutableMapping, RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer)
from google.protobuf.message import Message
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
    >>> protomsg = pybmix.proto.hierarchy_prior_pb2.NNIGPrior()
    >>> prior = pybmix.proto.hierarchy_prior_pb2.NNIGPrior.FixedValues(mean=1)
    >>> set_oneof_field("prior", protomsg, prior)
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


def get_field(msg, field: str):
    """Gets the field of an object, even if it is in a submessage

    Parameters
    ----------
    msg: google.protobuf.message
        a protobuf (object)
    field: str
        a field of this proto. If it is a nested field,
        then we adopt the syntax a joining the subfields separated by '.'
    """
    subfields = field.split(".")
    curr = msg
    try:
        for subfield in subfields:
            index = -1
            if subfield.endswith("]") and "[" in subfield[:-1]:
                parts = subfield[:-1].split("[")
                index = int(parts[1])
                subfield = parts[0]
            curr = _get_shallow_field(curr, subfield)
            if index >= 0 and index < len(curr):
                curr = curr[index]
        return curr
    except Exception as e:
        return None


def _get_shallow_field(msg, field: str):
    """Internal method to get the value of the field of the protobuf.

    This method doesn't go down the hierarchy and does not support the '.'
    syntax. Use get_field instead
    """
    if isinstance(msg, MutableMapping):
        try:
            return msg[field]

        except TypeError as e:
            return msg[int(field)]

    if hasattr(msg, field):
        return getattr(msg, field)

    return


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
