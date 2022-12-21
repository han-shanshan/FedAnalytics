import logging
from abc import abstractmethod
from .communication.base_com_manager import BaseCommunicationManager
from .communication.observer import Observer


class FedMLCommManager(Observer):
    def __init__(self, args, comm=None, rank=0, size=0, backend="MPI"):
        self.args = args
        self.size = size
        self.rank = int(rank)
        self.backend = backend
        self.comm = comm
        self.com_manager = None
        self.message_handler_dict = dict()
        self._init_manager()

    def register_comm_manager(self, comm_manager: BaseCommunicationManager):
        self.com_manager = comm_manager

    def run(self):
        self.register_message_receive_handlers()
        logging.info("running")
        self.com_manager.handle_receive_message()
        logging.info("finished...")

    def get_sender_id(self):
        return self.rank

    def receive_message(self, msg_type, msg_params) -> None:

        if msg_params.get_sender_id() == msg_params.get_receiver_id():
            logging.info("communication backend is alive (loop_forever, sender 0 to receiver 0)")
        else:
            logging.info(
                "receive_message. msg_type = %s, sender_id = %d, receiver_id = %d"
                % (str(msg_type), msg_params.get_sender_id(), msg_params.get_receiver_id())
            )
        try:
            handler_callback_func = self.message_handler_dict[msg_type]
            handler_callback_func(msg_params)
        except KeyError:
            raise Exception(
                "KeyError. msg_type = {}. Please check whether you launch the server or client with the correct args.rank".format(
                    msg_type
                )
            )

    def send_message(self, message):
        self.com_manager.send_message(message)

    def send_message_json(self, topic_name, json_message):
        self.com_manager.send_message_json(topic_name, json_message)

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        self.message_handler_dict[msg_type] = handler_callback_func

    def finish(self):
        logging.info("__finish")
        if self.backend == "MPI":
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort()

    def _init_manager(self):
        if self.backend == "MPI":
            from .communication.mpi.com_manager import MpiCommunicationManager

            self.com_manager = MpiCommunicationManager(self.comm, self.rank, self.size)
        else:
            if self.com_manager is None:
                raise Exception("no such backend: {}. Please check the comm_backend spelling.".format(self.backend))
            else:
                logging.info("using self-defined communication backend")

        self.com_manager.add_observer(self)
